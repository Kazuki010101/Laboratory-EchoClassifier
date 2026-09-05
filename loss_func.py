import math
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """Backward-compatible logit KD plus optional routing/content distillation."""
    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau,
                 route_weight=0.0, content_weight=0.0, teacher_layer=1, patch_size=16):
        super().__init__()
        if distillation_type not in {"none", "soft", "hard", "soft2", "soft3", "soft4"}:
            raise ValueError(distillation_type)
        self.base_criterion, self.teacher_model = base_criterion, teacher_model
        self.distillation_type, self.alpha, self.tau = distillation_type, alpha, tau
        self.route_weight, self.content_weight = route_weight, content_weight
        self.teacher_layer, self.patch_size = teacher_layer, patch_size
        self.last = {"base_loss": None, "kd_loss": None, "route_loss": None,
                     "content_loss": None, "t_entropy": None, "s_entropy": None,
                     "agree": None, "alpha": alpha, "tau": tau}

    @staticmethod
    def _fixed_projection(source_dim, target_dim, device, dtype):
        i = torch.arange(source_dim, device=device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(target_dim, device=device, dtype=torch.float32).unsqueeze(0)
        matrix = torch.cos((i + 0.5) * (j + 1.0) * math.pi / source_dim)
        return F.normalize(matrix, dim=0).to(dtype=dtype)

    def forward(self, inputs, student_output, labels):
        aux = None
        if isinstance(student_output, torch.Tensor):
            logits, logits_kd = student_output, None
        else:
            logits = student_output[0]
            logits_kd = student_output[1] if len(student_output) > 1 else None
            aux = student_output[2] if len(student_output) > 2 else None
        base_loss = self.base_criterion(logits, labels)
        kd_loss = route_loss = content_loss = None
        teacher_logits = teacher_route = teacher_evidence = None
        guided = aux is not None and aux.get("method") in {"teacher_skip", "tgec"}
        if self.distillation_type != "none" or guided:
            if self.teacher_model is None:
                raise ValueError("Teacher model is required for teacher-guided training")
            with torch.no_grad():
                if guided:
                    teacher_logits, teacher_route, teacher_evidence = self.teacher_model.forward_with_temporal_evidence(
                        inputs, self.teacher_layer, self.patch_size)
                else:
                    teacher_logits = self.teacher_model(inputs)
                if isinstance(teacher_logits, (tuple, list)):
                    teacher_logits = teacher_logits[0]
        loss = base_loss
        if self.distillation_type != "none":
            if logits_kd is None:
                raise ValueError("KD requires a second student logit tensor")
            if self.distillation_type == "soft":
                t = self.tau
                kd_loss = F.kl_div(F.log_softmax(logits_kd / t, dim=1),
                                   F.softmax(teacher_logits / t, dim=1),
                                   reduction="batchmean") * (t * t)
            elif self.distillation_type == "hard":
                kd_loss = F.cross_entropy(logits_kd, teacher_logits.argmax(1))
            else:
                raise NotImplementedError("This package supports soft/hard KD for new methods")
            loss = (1.0 - self.alpha) * base_loss + self.alpha * kd_loss
        if guided:
            route_loss = F.kl_div(F.log_softmax(aux["route_logits"], dim=1),
                                  teacher_route, reduction="batchmean")
            loss = loss + self.route_weight * route_loss
            if aux.get("summary_token") is not None and self.content_weight > 0:
                omitted = aux["omitted_mask"].unsqueeze(-1).to(teacher_evidence.dtype)
                target = (teacher_evidence * omitted).sum(1) / omitted.sum(1).clamp_min(1.0)
                target = target @ self._fixed_projection(target.shape[-1], aux["summary_token"].shape[-1],
                                                         target.device, target.dtype)
                content_loss = (1.0 - F.cosine_similarity(aux["summary_token"], target, dim=1)).mean()
                loss = loss + self.content_weight * content_loss
        with torch.no_grad():
            values = {"base_loss": float(base_loss), "kd_loss": None if kd_loss is None else float(kd_loss),
                      "route_loss": None if route_loss is None else float(route_loss),
                      "content_loss": None if content_loss is None else float(content_loss),
                      "t_entropy": None, "s_entropy": None, "agree": None,
                      "alpha": self.alpha, "tau": self.tau}
            if teacher_logits is not None and logits_kd is not None:
                tp, sp = F.softmax(teacher_logits, 1), F.softmax(logits_kd, 1)
                values.update(t_entropy=float(-(tp * tp.clamp_min(1e-12).log()).sum(1).mean()),
                              s_entropy=float(-(sp * sp.clamp_min(1e-12).log()).sum(1).mean()),
                              agree=float((teacher_logits.argmax(1) == logits_kd.argmax(1)).float().mean()))
            self.last.update(values)
        return loss
