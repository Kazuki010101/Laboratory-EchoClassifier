from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from selection_distillation_loss import (
    PatchSelectionDistillationLoss,
)
from teacher_patch_importance import (
    SenvtAttentionImportanceExtractor,
)


class TeacherGuidedDistillationLoss(nn.Module):
    """
    Combined loss for teacher-guided APS-PRC.

    Total loss:

        (1 - alpha) * classification loss
        + alpha * logit KD loss
        + selection_weight * patch selection loss

    Student training output:

        (
            classification_logits,
            distillation_logits,
            patch_score_logits,
        )
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        teacher_model: nn.Module,
        patch_size: int,
        stride: int,
        keep_ratio: float = 0.5,
        distillation_alpha: float = 0.7,
        distillation_temperature: float = 2.5,
        selection_weight: float = 1.0,
        selection_student_temperature: float = 1.0,
        selection_teacher_temperature: float = 1.0,
    ):
        super().__init__()

        if not 0.0 <= distillation_alpha <= 1.0:
            raise ValueError(
                "distillation_alpha must be in [0, 1]."
            )

        if distillation_temperature <= 0:
            raise ValueError(
                "distillation_temperature must be positive."
            )

        if selection_weight < 0:
            raise ValueError(
                "selection_weight must be non-negative."
            )

        self.base_criterion = base_criterion
        self.teacher_model = teacher_model

        self.patch_size = patch_size
        self.stride = stride

        self.distillation_alpha = (
            distillation_alpha
        )

        self.distillation_temperature = (
            distillation_temperature
        )

        self.selection_weight = selection_weight

        self.importance_extractor = (
            SenvtAttentionImportanceExtractor(
                teacher_model
            )
        )

        self.selection_criterion = (
            PatchSelectionDistillationLoss(
                keep_ratio=keep_ratio,
                student_temperature=(
                    selection_student_temperature
                ),
                teacher_temperature=(
                    selection_teacher_temperature
                ),
            )
        )

        self.last: Dict[str, float] = {}

    def close(self):
        self.importance_extractor.close()

    def forward(
        self,
        inputs: torch.Tensor,
        outputs,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(outputs, (tuple, list)):
            raise ValueError(
                "Teacher-guided APS student must return "
                "three tensors during training."
            )

        if len(outputs) != 3:
            raise ValueError(
                "Expected student output "
                "(classification_logits, "
                "distillation_logits, "
                "patch_score_logits), "
                f"but got {len(outputs)} outputs."
            )

        (
            classification_logits,
            distillation_logits,
            patch_score_logits,
        ) = outputs

        classification_loss = self.base_criterion(
            classification_logits,
            labels,
        )

        teacher_result = (
            self.importance_extractor.extract(
                inputs,
                student_patch_size=self.patch_size,
                student_stride=self.stride,
            )
        )

        teacher_logits = teacher_result[
            "teacher_logits"
        ]

        teacher_patch_importance = teacher_result[
            "teacher_patch_importance"
        ]

        temperature = (
            self.distillation_temperature
        )

        logit_distillation_loss = F.kl_div(
            F.log_softmax(
                distillation_logits / temperature,
                dim=1,
            ),
            F.softmax(
                teacher_logits / temperature,
                dim=1,
            ),
            reduction="batchmean",
        )

        logit_distillation_loss = (
            logit_distillation_loss
            * temperature
            * temperature
        )

        selection_loss = self.selection_criterion(
            patch_score_logits,
            teacher_patch_importance,
        )

        total_loss = (
            (1.0 - self.distillation_alpha)
            * classification_loss
            + self.distillation_alpha
            * logit_distillation_loss
            + self.selection_weight
            * selection_loss
        )

        with torch.no_grad():
            teacher_prediction = (
                teacher_logits.argmax(dim=1)
            )

            student_prediction = (
                distillation_logits.argmax(dim=1)
            )

            teacher_student_agreement = (
                teacher_prediction
                == student_prediction
            ).float().mean()

            selection_metrics = (
                self.selection_criterion
                .get_last_metrics()
            )

            self.last = {
                "base_loss": float(
                    classification_loss.detach().cpu()
                ),
                "kd_loss": float(
                    logit_distillation_loss
                    .detach()
                    .cpu()
                ),
                "selection_loss": float(
                    selection_loss.detach().cpu()
                ),
                "total_loss": float(
                    total_loss.detach().cpu()
                ),
                "teacher_student_agreement": float(
                    teacher_student_agreement
                    .detach()
                    .cpu()
                ),
                "distillation_alpha": float(
                    self.distillation_alpha
                ),
                "distillation_temperature": float(
                    self.distillation_temperature
                ),
                "selection_weight": float(
                    self.selection_weight
                ),
            }

            for key, value in selection_metrics.items():
                self.last[
                    f"selection_{key}"
                ] = value

        return total_loss

    def get_last_metrics(self) -> Dict[str, float]:
        return dict(self.last)
