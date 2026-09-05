import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Existing PRC-compatible patch embedding.

    Input:
        x: [B, C, L]

    Output:
        patches: [B, D, T]

    D equals patch_size.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        stride: int,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=patch_size,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x)


class TeacherGuidedAdaptiveReservoirNetwork(nn.Module):
    """
    Full-rank PRC reservoir with adaptive patch selection.

    This class intentionally excludes:

    - low-rank reservoir factorization
    - update gate
    - multiplication of state updates by patch scores

    The only difference from the original PRC reservoir is that
    only top-k selected patches are processed.
    """

    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        keep_ratio: float = 0.5,
        spectral_radius: float = 0.9,
    ):
        super().__init__()

        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(
                "keep_ratio must be in the range (0, 1]."
            )

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.keep_ratio = keep_ratio
        self.spectral_radius = spectral_radius

        # Same full-rank frozen reservoir as the original PRC.
        self.W_reservoir = nn.Parameter(
            torch.rand(
                reservoir_size,
                reservoir_size,
            ) - 0.5,
            requires_grad=False,
        )

        self.W_input = nn.Parameter(
            torch.rand(
                reservoir_size,
                input_size,
            ) - 0.5,
            requires_grad=False,
        )

        # Scale to the requested spectral radius.
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(
                self.W_reservoir
            )

            maximum_eigenvalue = torch.max(
                torch.abs(eigenvalues)
            )

            if maximum_eigenvalue <= 0:
                raise RuntimeError(
                    "Maximum reservoir eigenvalue is not positive."
                )

            self.W_reservoir.mul_(
                spectral_radius / maximum_eigenvalue
            )

        # Trainable APS scorer.
        # One scalar logit is produced for every embedded patch.
        self.patch_scorer = nn.Linear(
            input_size,
            1,
        )

    def _calculate_keep_count(
        self,
        sequence_length: int,
    ) -> int:
        keep_count = int(
            math.ceil(
                sequence_length * self.keep_ratio
            )
        )

        return max(
            1,
            min(sequence_length, keep_count),
        )

    def select_patches(
        self,
        x: torch.Tensor,
        ranking_scores: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x:
                Embedded patches with shape [B, D, T].

            ranking_scores:
                Optional external scores with shape [B, T].
                This is intended only for teacher-oracle analysis.
                Normal APS inference does not use the teacher.

        Returns:
            selected_patches:
                [B, K, D]

            selection:
                Dictionary containing scores, indices, and mask.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with 3 dimensions, got {x.shape}."
            )

        patches = x.transpose(1, 2)
        batch_size, sequence_length, dimension = (
            patches.shape
        )

        patch_score_logits = self.patch_scorer(
            patches
        ).squeeze(-1)

        patch_probabilities = torch.sigmoid(
            patch_score_logits
        )

        if ranking_scores is None:
            scores_for_ranking = patch_score_logits
        else:
            if ranking_scores.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "ranking_scores shape mismatch: "
                    f"expected {(batch_size, sequence_length)}, "
                    f"got {tuple(ranking_scores.shape)}."
                )

            scores_for_ranking = ranking_scores

        keep_count = self._calculate_keep_count(
            sequence_length
        )

        if keep_count < sequence_length:
            topk_indices = torch.topk(
                scores_for_ranking,
                k=keep_count,
                dim=1,
                largest=True,
                sorted=False,
            ).indices

            # Reservoir processing must retain temporal order.
            selected_indices = torch.sort(
                topk_indices,
                dim=1,
            ).values
        else:
            selected_indices = torch.arange(
                sequence_length,
                device=x.device,
            ).unsqueeze(0).expand(
                batch_size,
                -1,
            )

        gather_indices = selected_indices.unsqueeze(
            -1
        ).expand(
            -1,
            -1,
            dimension,
        )

        selected_patches = torch.gather(
            patches,
            dim=1,
            index=gather_indices,
        )

        selected_mask = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        )

        selected_mask.scatter_(
            dim=1,
            index=selected_indices,
            value=True,
        )

        selection = {
            "patch_score_logits": patch_score_logits,
            "patch_probabilities": patch_probabilities,
            "selected_indices": selected_indices,
            "selected_mask": selected_mask,
            "keep_count": keep_count,
            "total_patch_count": sequence_length,
        }

        return selected_patches, selection

    def forward(
        self,
        x: torch.Tensor,
        cls_token: torch.Tensor,
        distillation_token: torch.Tensor,
        ranking_scores: Optional[torch.Tensor] = None,
    ):
        selected_patches, selection = (
            self.select_patches(
                x,
                ranking_scores=ranking_scores,
            )
        )

        batch_size = x.shape[0]
        selected_patch_count = selected_patches.shape[1]

        hidden_state = torch.zeros(
            batch_size,
            self.reservoir_size,
            dtype=x.dtype,
            device=x.device,
        )

        # Pure PRC recurrence.
        # APS scores are not multiplied into the state update.
        for patch_index in range(selected_patch_count):
            current_patch = selected_patches[
                :,
                patch_index,
                :,
            ]

            hidden_state = torch.tanh(
                torch.matmul(
                    hidden_state,
                    self.W_reservoir,
                )
                + torch.matmul(
                    current_patch,
                    self.W_input.T,
                )
            )

        class_state = torch.tanh(
            torch.matmul(
                hidden_state,
                self.W_reservoir,
            )
            + torch.matmul(
                cls_token,
                self.W_input.T,
            )
        )

        distillation_state = torch.tanh(
            torch.matmul(
                hidden_state,
                self.W_reservoir,
            )
            + torch.matmul(
                distillation_token,
                self.W_input.T,
            )
        )

        return (
            class_state,
            distillation_state,
            selection,
        )


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
    ):
        super().__init__()

        self.fc = nn.Linear(
            input_size,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(x)


class PatchReservoir(nn.Module):
    """
    Full-rank PRC with teacher-guided adaptive patch selection.

    Standard training output:
        (classification_logits, distillation_logits)

    Standard evaluation output:
        averaged logits

    Analysis output with return_selection=True:
        dictionary containing logits and selection information
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        stride: int,
        reservoir_size: int,
        patch_keep_ratio: float,
        num_classes: int,
        spectral_radius: float = 0.9,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.patch_keep_ratio = patch_keep_ratio

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
        )

        self.reservoir_network = (
            TeacherGuidedAdaptiveReservoirNetwork(
                input_size=patch_size,
                reservoir_size=reservoir_size,
                keep_ratio=patch_keep_ratio,
                spectral_radius=spectral_radius,
            )
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, patch_size)
        )

        self.distillation_token = nn.Parameter(
            torch.zeros(1, patch_size)
        )

        self.classification_head = ClassificationHead(
            input_size=reservoir_size,
            num_classes=num_classes,
        )

        self.distillation_head = ClassificationHead(
            input_size=reservoir_size,
            num_classes=num_classes,
        )

        self._last_selection = None

    def forward(
        self,
        x: torch.Tensor,
        return_selection: bool = False,
        ranking_scores: Optional[torch.Tensor] = None,
    ):
        embedded_patches = self.patch_embed(x)

        class_tokens = self.cls_token.expand(
            embedded_patches.shape[0],
            -1,
        )

        distillation_tokens = (
            self.distillation_token.expand(
                embedded_patches.shape[0],
                -1,
            )
        )

        (
            class_state,
            distillation_state,
            selection,
        ) = self.reservoir_network(
            embedded_patches,
            class_tokens,
            distillation_tokens,
            ranking_scores=ranking_scores,
        )

        classification_logits = (
            self.classification_head(
                class_state
            )
        )

        distillation_logits = (
            self.distillation_head(
                distillation_state
            )
        )

        self._last_selection = selection

        averaged_logits = (
            classification_logits
            + distillation_logits
        ) / 2.0

        if return_selection:
            return {
                "logits": averaged_logits,
                "classification_logits": (
                    classification_logits
                ),
                "distillation_logits": (
                    distillation_logits
                ),
                "patch_score_logits": selection[
                    "patch_score_logits"
                ],
                "patch_probabilities": selection[
                    "patch_probabilities"
                ],
                "selected_indices": selection[
                    "selected_indices"
                ],
                "selected_mask": selection[
                    "selected_mask"
                ],
                "keep_count": selection[
                    "keep_count"
                ],
                "total_patch_count": selection[
                    "total_patch_count"
                ],
                "patch_size": self.patch_size,
                "stride": self.stride,
            }

        if self.training:
            return (
                classification_logits,
                distillation_logits,
                selection["patch_score_logits"],
            )

        return averaged_logits

    def get_selection_info(
        self,
        detach: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if self._last_selection is None:
            raise RuntimeError(
                "No selection information is available. "
                "Run forward() first."
            )

        if not detach:
            return self._last_selection

        result = {}

        for key, value in self._last_selection.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach()
            else:
                result[key] = value

        return result

    def get_routing_stats(self) -> Dict[str, float]:
        if self._last_selection is None:
            return {}

        selected_mask = self._last_selection[
            "selected_mask"
        ]

        return {
            "hard_keep_ratio": float(
                selected_mask.float().mean().detach().cpu()
            ),
            "selected_patches_mean": float(
                selected_mask.sum(dim=1)
                .float()
                .mean()
                .detach()
                .cpu()
            ),
        }
