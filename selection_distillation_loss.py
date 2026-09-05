import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSelectionDistillationLoss(nn.Module):
    """
    Distill SENvT patch importance into APS patch scores.

    Student input:
        APS score logits [B, T]

    Teacher input:
        Non-negative SENvT patch importance [B, T]

    Loss:
        KL(
            teacher patch distribution
            ||
            student patch distribution
        )
    """

    def __init__(
        self,
        keep_ratio: float = 0.5,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        epsilon: float = 1e-12,
    ):
        super().__init__()

        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(
                "keep_ratio must be in the range (0, 1]."
            )

        if student_temperature <= 0:
            raise ValueError(
                "student_temperature must be positive."
            )

        if teacher_temperature <= 0:
            raise ValueError(
                "teacher_temperature must be positive."
            )

        self.keep_ratio = keep_ratio
        self.student_temperature = (
            student_temperature
        )
        self.teacher_temperature = (
            teacher_temperature
        )
        self.epsilon = epsilon

        self.last_metrics: Dict[str, float] = {}

    def _teacher_distribution(
        self,
        teacher_importance: torch.Tensor,
    ) -> torch.Tensor:
        teacher_importance = (
            teacher_importance.clamp_min(0.0)
        )

        teacher_importance = (
            teacher_importance
            / teacher_importance.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(self.epsilon)
        )

        # Convert probabilities back to log-space before
        # applying teacher temperature.
        teacher_log_scores = torch.log(
            teacher_importance.clamp_min(
                self.epsilon
            )
        )

        return F.softmax(
            teacher_log_scores
            / self.teacher_temperature,
            dim=1,
        )

    def _calculate_metrics(
        self,
        student_score_logits: torch.Tensor,
        teacher_distribution: torch.Tensor,
    ):
        patch_count = student_score_logits.shape[1]

        keep_count = int(
            math.ceil(
                patch_count * self.keep_ratio
            )
        )

        keep_count = max(
            1,
            min(patch_count, keep_count),
        )

        student_indices = torch.topk(
            student_score_logits,
            k=keep_count,
            dim=1,
        ).indices

        teacher_indices = torch.topk(
            teacher_distribution,
            k=keep_count,
            dim=1,
        ).indices

        student_mask = torch.zeros_like(
            student_score_logits,
            dtype=torch.bool,
        )

        teacher_mask = torch.zeros_like(
            student_score_logits,
            dtype=torch.bool,
        )

        student_mask.scatter_(
            dim=1,
            index=student_indices,
            value=True,
        )

        teacher_mask.scatter_(
            dim=1,
            index=teacher_indices,
            value=True,
        )

        intersection = (
            student_mask
            & teacher_mask
        ).sum(dim=1).float()

        union = (
            student_mask
            | teacher_mask
        ).sum(dim=1).float()

        topk_overlap = (
            intersection / keep_count
        ).mean()

        jaccard = (
            intersection
            / union.clamp_min(1.0)
        ).mean()

        retained_teacher_mass = torch.gather(
            teacher_distribution,
            dim=1,
            index=student_indices,
        ).sum(dim=1).mean()

        student_distribution = F.softmax(
            student_score_logits
            / self.student_temperature,
            dim=1,
        )

        student_entropy = (
            -student_distribution
            * torch.log(
                student_distribution.clamp_min(
                    self.epsilon
                )
            )
        ).sum(dim=1).mean()

        teacher_entropy = (
            -teacher_distribution
            * torch.log(
                teacher_distribution.clamp_min(
                    self.epsilon
                )
            )
        ).sum(dim=1).mean()

        self.last_metrics = {
            "patch_count": float(patch_count),
            "keep_count": float(keep_count),
            "topk_overlap": float(
                topk_overlap.detach().cpu()
            ),
            "jaccard": float(
                jaccard.detach().cpu()
            ),
            "retained_teacher_mass": float(
                retained_teacher_mass.detach().cpu()
            ),
            "student_entropy": float(
                student_entropy.detach().cpu()
            ),
            "teacher_entropy": float(
                teacher_entropy.detach().cpu()
            ),
        }

    def forward(
        self,
        student_score_logits: torch.Tensor,
        teacher_importance: torch.Tensor,
    ) -> torch.Tensor:
        if student_score_logits.ndim != 2:
            raise ValueError(
                "student_score_logits must have shape "
                f"[B, T], got {student_score_logits.shape}."
            )

        if teacher_importance.ndim != 2:
            raise ValueError(
                "teacher_importance must have shape "
                f"[B, T], got {teacher_importance.shape}."
            )

        if (
            student_score_logits.shape
            != teacher_importance.shape
        ):
            raise ValueError(
                "Student and teacher patch shapes differ: "
                f"{student_score_logits.shape} vs "
                f"{teacher_importance.shape}."
            )

        teacher_distribution = (
            self._teacher_distribution(
                teacher_importance
            )
        )

        student_log_distribution = F.log_softmax(
            student_score_logits
            / self.student_temperature,
            dim=1,
        )

        selection_loss = F.kl_div(
            student_log_distribution,
            teacher_distribution,
            reduction="batchmean",
        )

        selection_loss = (
            selection_loss
            * self.student_temperature
            * self.student_temperature
        )

        with torch.no_grad():
            self._calculate_metrics(
                student_score_logits,
                teacher_distribution,
            )

        return selection_loss

    def get_last_metrics(self) -> Dict[str, float]:
        return dict(self.last_metrics)
