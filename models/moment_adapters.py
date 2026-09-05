from typing import Union, Dict, Any, Iterable, Optional, Tuple, Callable
import os
import inspect
import torch
import torch.nn as nn

try:
    from momentfm import MOMENTPipeline
except Exception as e:
    raise ImportError("momentfm not found. Please `pip install momentfm`.") from e


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    v = os.environ.get(name, None)
    return v if (v is not None and v.strip() != "") else default


FORCED_KEY = _env_str("MOMENT_INPUT_KEY", None)
FORCED_TRANSPOSE = _env_flag("MOMENT_TRANSPOSE", False)


def _extract_logits(output: Union[torch.Tensor, Dict[str, Any], object]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for k in ("logits", "pred", "y_hat", "output", "logit"):
            v = output.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("Could not find logits tensor in dict output.")
    for k in ("logits", "pred", "y_hat", "output", "logit"):
        if hasattr(output, k):
            v = getattr(output, k)
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(output, "to_dict"):
        d = output.to_dict()
        if isinstance(d, dict):
            for k in ("logits", "pred", "y_hat", "output", "logit"):
                v = d.get(k, None)
                if isinstance(v, torch.Tensor):
                    return v
    raise TypeError(f"Unsupported output type for logits extraction: {type(output)}")


def _iter_modules(obj: object, visited: set) -> Iterable[nn.Module]:
    if id(obj) in visited:
        return
    visited.add(id(obj))
    if isinstance(obj, nn.Module):
        yield obj
    for _, v in getattr(obj, "__dict__", {}).items():
        if isinstance(v, nn.Module):
            yield v
            yield from _iter_modules(v, visited)
        elif isinstance(v, (list, tuple, set)):
            for it in v:
                if isinstance(it, nn.Module):
                    yield it
                    yield from _iter_modules(it, visited)
        elif isinstance(v, dict):
            for it in v.values():
                if isinstance(it, nn.Module):
                    yield it
                    yield from _iter_modules(it, visited)


def _pick_largest_module(root: object) -> Optional[nn.Module]:
    best, best_n = None, -1
    for m in _iter_modules(root, set()):
        n = sum(p.numel() for p in m.parameters(recurse=True))
        if n > best_n:
            best, best_n = m, n
    return best


def _maybe_transpose_bc_l(x: torch.Tensor) -> torch.Tensor:
    if FORCED_TRANSPOSE:
        return x.transpose(1, 2).contiguous() if x.ndim == 3 else x
    if x.ndim == 3 and x.size(1) <= 4 and x.size(2) > 8:
        return x
    if x.ndim == 3 and x.size(2) <= 4 and x.size(1) > 8:
        return x.transpose(1, 2).contiguous()
    return x


def _signature_of(fn: Callable) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "(signature unknown)"


def _list_callables(obj: object) -> Dict[str, str]:
    out = {}
    names = set(dir(obj))
    for n in sorted(names):
        try:
            a = getattr(obj, n)
        except Exception:
            continue
        if callable(a):
            out[n] = _signature_of(a)
    return out


def _set_candidate_attrs(target: object, x: torch.Tensor, keys: Tuple[str, ...]) -> None:
    for k in keys:
        try:
            setattr(target, k, x)
        except Exception:
            pass


def _call_with_signatures(fn, x: torch.Tensor) -> Any:
    keys = ("x_enc", "x", "inputs", "signals", "data")
    if FORCED_KEY is not None:
        try:
            return fn(**{FORCED_KEY: x})
        except TypeError:
            pass
    for k in keys:
        if k == FORCED_KEY:
            continue
        try:
            return fn(**{k: x})
        except TypeError:
            pass
    try:
        return fn(x)
    except TypeError:
        pass
    for kd in ("x_enc", "x"):
        try:
            return fn({kd: x})
        except TypeError:
            pass
    raise TypeError("No compatible call signature worked for the given function.")


def _call_moment_any(core: Optional[nn.Module], pipe, x: torch.Tensor):
    x = _maybe_transpose_bc_l(x)
    if core is not None:
        try:
            return _call_with_signatures(core, x)
        except TypeError:
            _set_candidate_attrs(core, x, ("x_enc", "x", "inputs", "signals", "data"))
            for cand in ("forward", "__call__"):
                fn = getattr(core, cand, None)
                if callable(fn):
                    try:
                        return fn()
                    except TypeError:
                        pass
    for attr in ("classify", "predict", "forward", "__call__", "run"):
        fn = getattr(pipe, attr, None)
        if callable(fn):
            try:
                return _call_with_signatures(fn, x)
            except TypeError:
                pass
    _set_candidate_attrs(pipe, x, ("x_enc", "x", "inputs", "signals", "data"))
    for cand in ("forward", "__call__", "run"):
        fn = getattr(pipe, cand, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                pass
    callables = _list_callables(pipe)
    core_callables = _list_callables(core) if core is not None else {}
    raise TypeError(
        "Could not invoke MOMENT model with any known calling convention.\n"
        f"- Forced key: {FORCED_KEY}, Forced transpose: {FORCED_TRANSPOSE}\n"
        f"- Pipe type: {type(pipe).__name__}\n"
        f"- Core type: {type(core).__name__ if core is not None else None}\n"
        f"- Pipe callables: {callables}\n"
        f"- Core callables: {core_callables}"
    )


class MomentTeacherAdapter(nn.Module):
    def __init__(self, hub_id: str, n_channels: int, num_class: int, device: torch.device):
        super().__init__()
        self.device = device
        self.pipe = MOMENTPipeline.from_pretrained(
            hub_id,
            model_kwargs={"task_name": "classification", "n_channels": n_channels, "num_class": num_class},
        )
        self.pipe.init()
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        core = _pick_largest_module(self.pipe)
        self.core = core if isinstance(core, nn.Module) else None
        if self.core is not None:
            self.core.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        out = _call_moment_any(self.core, self.pipe, x)
        return _extract_logits(out)


class MomentStudentAdapter(nn.Module):
    def __init__(self, hub_id: str, n_channels: int, num_class: int, device: torch.device):
        super().__init__()
        self.device = device
        self.pipe = MOMENTPipeline.from_pretrained(
            hub_id,
            model_kwargs={"task_name": "classification", "n_channels": n_channels, "num_class": num_class},
        )
        self.pipe.init()
        core = _pick_largest_module(self.pipe)
        if not isinstance(core, nn.Module):
            raise RuntimeError(
                "Could not locate a trainable nn.Module inside MOMENTPipeline. "
                "Please verify your momentfm version or adapt to upstream API changes."
            )
        self.core = core.to(self.device)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.to(self.device, non_blocking=True)
        out = _call_moment_any(self.core, self.pipe, x)
        logits = _extract_logits(out)
        if self.training:
            return logits, logits
        return logits
