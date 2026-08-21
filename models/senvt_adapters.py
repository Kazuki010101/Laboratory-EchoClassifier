# senvt_adapters.py
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from models import senvt 


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _choose_state_dict_like(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt  
    raise ValueError("Checkpoint does not contain a recognizable state_dict/model.")


# def _strip_prefix_if_present(state_dict, prefix: str):
#     if not any(k.startswith(prefix) for k in state_dict.keys()):
#         return state_dict
#     return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }

def _safe_load_partial(model: nn.Module, src_state: Dict[str, torch.Tensor], verbose: bool = True) -> None:
    dst_state = model.state_dict()

    def _match_count(sd): 
        return sum(1 for k in sd.keys() if k in dst_state)

    cand = []
    cand.append(src_state)
    cand.append(_strip_prefix_if_present(src_state, "module."))
    cand.append(_strip_prefix_if_present(src_state, "core."))
    cand.append(_strip_prefix_if_present(_strip_prefix_if_present(src_state, "module."), "core."))
    src_state = max(cand, key=_match_count)

    filtered = {}
    skipped_shape, skipped_missing = [], []
    for k, v in src_state.items():
        if k in dst_state:
            if v.shape == dst_state[k].shape:
                filtered[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(dst_state[k].shape)))
        else:
            skipped_missing.append(k)

    if len(filtered) == 0:
        raise RuntimeError(
            "[SENvT CKPT] No parameters were loaded. "
            "Checkpoint format or prefix may be wrong."
        )
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    
    if verbose:
        print(f"[SENvT CKPT] Loaded params: {len(filtered)}")
        if skipped_shape:
            print(f"[SENvT CKPT] Skipped (shape mismatch): {len(skipped_shape)}")
            for k, s_src, s_dst in skipped_shape[:10]:
                print(f"  - {k}: src{s_src} != dst{s_dst}")
            if len(skipped_shape) > 10:
                print(f"  ... and {len(skipped_shape) - 10} more")
        if skipped_missing:
            print(f"[SENvT CKPT] Skipped (not in dst): {len(skipped_missing)} (e.g., {skipped_missing[:5]})")
        if missing_keys:
            print(f"[SENvT CKPT] Missing in ckpt (kept init): {len(missing_keys)} (e.g., {missing_keys[:5]})")
        if unexpected_keys:
            print(f"[SENvT CKPT] Unexpected in ckpt: {len(unexpected_keys)} (e.g., {unexpected_keys[:5]})")



class SenvtTeacherAdapter(nn.Module):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        window_size: int,
        in_chans: int,
        ckpt_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        verbose_ckpt: bool = True
    ):
        super().__init__()

        if variant == "XS":
            core = senvt.XS(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        elif variant == "S":
            core = senvt.S(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        elif variant == "B":
            core = senvt.B(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        elif variant == "L":
            core = senvt.L(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        else:
            raise ValueError(f"Unknown SENvT variant: {variant}")

        if ckpt_path is not None and ckpt_path != "":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            src = _choose_state_dict_like(ckpt)
            src = _strip_prefix_if_present(src, prefix="module.")
            _safe_load_partial(core, src, verbose=verbose_ckpt)

        self.core = core.to(device)
        self.device = device

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        logits = self.core(x)
    
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
    
        return logits


class SenvtStudentAdapter(nn.Module):
    def __init__(
        self,
        variant: str,
        num_classes: int,
        window_size: int,
        in_chans: int,
        device: torch.device = torch.device("cuda"),
        ckpt_path: Optional[str] = None,
        verbose_ckpt: bool = True,
    ):
        super().__init__()

        if variant == "XS":
            core = senvt.XS(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        elif variant == "S":
            core = senvt.S(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        elif variant == "B":
            core = senvt.B(num_classes=num_classes, window_size=window_size, in_chans=in_chans)
        else:
            raise ValueError(f"Unknown SENvT student variant: {variant}")

        if ckpt_path is not None and ckpt_path != "":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            src = _choose_state_dict_like(ckpt)
            src = _strip_prefix_if_present(src, prefix="module.")
            _safe_load_partial(core, src, verbose=verbose_ckpt)

        self.core = core.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        x = x.to(self.device, non_blocking=True)
        logits = self.core(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if self.training:
            return logits, logits

        return logits
