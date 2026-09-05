from typing import Dict, Optional

import torch
import torch.nn as nn


class SenvtAttentionImportanceExtractor:
    """
    Extract patch importance from the final SENvT attention layer.

    Teacher importance:
        Mean attention from the CLS token to each signal token,
        averaged across attention heads.

    The SENvT teacher is used only during training or analysis.
    It is not required during student inference.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
    ):
        self.teacher_model = teacher_model
        self.teacher_core = self._find_teacher_core(
            teacher_model
        )

        if not hasattr(self.teacher_core, "layers"):
            raise AttributeError(
                "SENvT core does not have encoder layers."
            )

        if len(self.teacher_core.layers) == 0:
            raise RuntimeError(
                "SENvT core has no encoder layers."
            )

        final_layer = self.teacher_core.layers[-1]

        if not hasattr(final_layer, "attention"):
            raise AttributeError(
                "Final SENvT layer has no attention module."
            )

        self._captured_attention = None

        self._hook_handle = (
            final_layer.attention.register_forward_hook(
                self._attention_hook
            )
        )

    @staticmethod
    def _find_teacher_core(
        teacher_model: nn.Module,
    ) -> nn.Module:
        """
        Supports:

        - raw SENvT model
        - SenvtTeacherAdapter
        - DistributedDataParallel wrapper
        """

        model = teacher_model

        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "core"):
            model = model.core

        return model

    def _attention_hook(
        self,
        module: nn.Module,
        inputs,
        output,
    ):
        del module
        del inputs

        if not isinstance(output, (tuple, list)):
            raise RuntimeError(
                "SENvT attention output is expected to be "
                "(features, attention)."
            )

        if len(output) < 2:
            raise RuntimeError(
                "SENvT attention output does not contain "
                "an attention tensor."
            )

        self._captured_attention = output[1]

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ):
        del exception_type
        del exception_value
        del traceback

        self.close()

    @staticmethod
    def _normalize_importance(
        importance: torch.Tensor,
    ) -> torch.Tensor:
        importance = importance.clamp_min(0.0)

        denominator = importance.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

        return importance / denominator

    def _expand_teacher_tokens_to_time(
        self,
        token_importance: torch.Tensor,
        input_length: int,
    ) -> torch.Tensor:
        patch_embed = self.teacher_core.patch_embed

        teacher_patch_size = int(
            patch_embed.patch_size
        )

        if teacher_patch_size <= 0:
            raise ValueError(
                "Teacher patch size must be positive."
            )

        if teacher_patch_size == 1:
            time_importance = token_importance
        else:
            time_importance = (
                token_importance.repeat_interleave(
                    teacher_patch_size,
                    dim=1,
                )
            )

        if time_importance.shape[1] < input_length:
            missing_length = (
                input_length
                - time_importance.shape[1]
            )

            padding = torch.zeros(
                time_importance.shape[0],
                missing_length,
                dtype=time_importance.dtype,
                device=time_importance.device,
            )

            time_importance = torch.cat(
                [time_importance, padding],
                dim=1,
            )

        return time_importance[:, :input_length]

    @staticmethod
    def _aggregate_for_student(
        time_importance: torch.Tensor,
        student_patch_size: int,
        student_stride: int,
    ) -> torch.Tensor:
        if student_patch_size <= 0:
            raise ValueError(
                "student_patch_size must be positive."
            )

        if student_stride <= 0:
            raise ValueError(
                "student_stride must be positive."
            )

        if time_importance.shape[1] < student_patch_size:
            raise ValueError(
                "Student patch size is larger than "
                "the input sequence."
            )

        patch_windows = time_importance.unfold(
            dimension=1,
            size=student_patch_size,
            step=student_stride,
        )

        # Sum is used so that the score represents the total
        # teacher importance contained within each PRC patch.
        patch_importance = patch_windows.sum(
            dim=-1
        )

        return patch_importance

    @torch.no_grad()
    def extract(
        self,
        inputs: torch.Tensor,
        student_patch_size: int,
        student_stride: Optional[int] = None,
        return_full_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if student_stride is None:
            student_stride = student_patch_size

        self._captured_attention = None

        self.teacher_model.eval()

        teacher_logits = self.teacher_model(inputs)

        if isinstance(teacher_logits, (tuple, list)):
            teacher_logits = teacher_logits[0]

        if self._captured_attention is None:
            raise RuntimeError(
                "No SENvT attention tensor was captured."
            )

        attention = self._captured_attention

        if attention.ndim != 4:
            raise RuntimeError(
                "Expected attention shape [B, H, N, N], "
                f"got {tuple(attention.shape)}."
            )

        # [B, H, N, N]
        # CLS query is position 0.
        # Signal tokens begin at position 1.
        cls_to_signal_attention = attention[
            :,
            :,
            0,
            1:,
        ]

        # Average over attention heads.
        teacher_token_importance = (
            cls_to_signal_attention.mean(dim=1)
        )

        teacher_token_importance = (
            self._normalize_importance(
                teacher_token_importance
            )
        )

        input_length = int(inputs.shape[-1])

        teacher_time_importance = (
            self._expand_teacher_tokens_to_time(
                teacher_token_importance,
                input_length=input_length,
            )
        )

        teacher_time_importance = (
            self._normalize_importance(
                teacher_time_importance
            )
        )

        teacher_patch_importance = (
            self._aggregate_for_student(
                teacher_time_importance,
                student_patch_size=student_patch_size,
                student_stride=student_stride,
            )
        )

        teacher_patch_importance = (
            self._normalize_importance(
                teacher_patch_importance
            )
        )

        result = {
            "teacher_logits": teacher_logits,
            "teacher_prediction": (
                teacher_logits.argmax(dim=1)
            ),
            "teacher_token_importance": (
                teacher_token_importance
            ),
            "teacher_time_importance": (
                teacher_time_importance
            ),
            "teacher_patch_importance": (
                teacher_patch_importance
            ),
        }

        if return_full_attention:
            result["full_attention"] = attention

        return result
