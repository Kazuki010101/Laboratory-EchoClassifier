import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Non-overlapping 1D patch embedding.

    Input:
        x: [B, C, L]

    Output:
        patches: [B, T, D]

    D is equal to patch_size in the current implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 32,
        stride: Optional[int] = None,
    ):
        super().__init__()

        if stride is None:
            stride = patch_size

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = patch_size

        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # [B, D, T]
        x = x.transpose(1, 2)     # [B, T, D]
        return x


class StateInnovationRouter(nn.Module):
    """
    Estimate whether the current patch contains information that is
    novel relative to the current low-rank reservoir state.

    Patch representation:
        u_t: [B, D]

    Current low-rank reservoir state:
        q_{t-1}: [B, R]

    Patch is first projected into the same R-dimensional space:
        e_t = input_projection(u_t)

    Router features:
        e_t
        q_{t-1}
        |e_t - q_{t-1}|
        e_t * q_{t-1}
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_rank: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, reservoir_rank),
            nn.LayerNorm(reservoir_rank),
            nn.Tanh(),
        )

        router_input_dim = reservoir_rank * 4

        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch: torch.Tensor,
        low_rank_state: torch.Tensor,
    ):
        """
        Args:
            patch:
                [B, D]

            low_rank_state:
                [B, R]

        Returns:
            route_probability:
                [B, 1]

            patch_state:
                [B, R]

            innovation_distance:
                [B, 1]

            innovation_cosine:
                [B, 1]
        """

        patch_state = self.input_projection(patch)

        difference = patch_state - low_rank_state

        distance = torch.linalg.vector_norm(
            difference,
            ord=2,
            dim=-1,
            keepdim=True,
        ) / math.sqrt(patch_state.shape[-1])

        cosine_similarity = F.cosine_similarity(
            patch_state,
            low_rank_state,
            dim=-1,
            eps=1e-8,
        ).unsqueeze(-1)

        cosine_novelty = 1.0 - cosine_similarity

        router_features = torch.cat(
            [
                patch_state,
                low_rank_state,
                torch.abs(difference),
                patch_state * low_rank_state,
            ],
            dim=-1,
        )

        route_logit = self.router(router_features)

        # Distance and cosine novelty are added as explicit
        # state-innovation signals.
        route_logit = route_logit + distance + cosine_novelty

        route_probability = torch.sigmoid(route_logit)

        return (
            route_probability,
            patch_state,
            distance,
            cosine_novelty,
        )


class StateInnovationLowRankGatedReservoirNetwork(nn.Module):
    """
    State Innovation Routed Low-Rank Gated Reservoir.

    During training:
        - A hard binary decision is used in the forward pass.
        - Straight-through gradients are used for the router.
        - All candidate states are calculated to preserve stable gradients.

    During evaluation:
        - Candidate reservoir updates are calculated only for samples
          selected by the router.
        - With batch size 1, an unselected patch avoids the full LRGR
          state-update calculation.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        reservoir_rank: int,
        router_hidden_dim: int = 64,
        routing_threshold: float = 0.5,
        target_keep_ratio: float = 0.5,
        minimum_keep_patches: int = 1,
        collect_routing_stats: bool = True,
    ):
        super().__init__()

        if reservoir_rank <= 0:
            raise ValueError("reservoir_rank must be greater than zero.")

        if reservoir_rank > reservoir_size:
            raise ValueError(
                "reservoir_rank must not exceed reservoir_size."
            )

        if not 0.0 < routing_threshold < 1.0:
            raise ValueError(
                "routing_threshold must be between 0 and 1."
            )

        if not 0.0 < target_keep_ratio <= 1.0:
            raise ValueError(
                "target_keep_ratio must be in the range (0, 1]."
            )

        if minimum_keep_patches < 1:
            raise ValueError(
                "minimum_keep_patches must be at least 1."
            )

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.reservoir_rank = reservoir_rank

        self.routing_threshold = routing_threshold
        self.target_keep_ratio = target_keep_ratio
        self.minimum_keep_patches = minimum_keep_patches
        self.collect_routing_stats = collect_routing_stats
        # Fixed low-rank reservoir matrices.
        self.U = nn.Parameter(
            torch.empty(reservoir_size, reservoir_rank),
            requires_grad=False,
        )

        self.V = nn.Parameter(
            torch.empty(reservoir_rank, reservoir_size),
            requires_grad=False,
        )

        self.W_input = nn.Parameter(
            torch.empty(reservoir_size, input_dim),
            requires_grad=False,
        )

        nn.init.normal_(self.U, mean=0.0, std=0.02)
        nn.init.normal_(self.V, mean=0.0, std=0.02)
        nn.init.normal_(self.W_input, mean=0.0, std=0.02)

        self.router = StateInnovationRouter(
            input_dim=input_dim,
            reservoir_rank=reservoir_rank,
            hidden_dim=router_hidden_dim,
        )

        self.update_gate = nn.Linear(input_dim, 1)

        self.state_norm = nn.LayerNorm(reservoir_size)

        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_routing_stats: Dict[str, float] = {}

    def _candidate_state(
        self,
        low_rank_state: torch.Tensor,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_{t-1} V + u_t W_in^T

        low_rank_state:
            [B, R]

        patch:
            [B, D]
        """

        recurrent_term = torch.matmul(
            low_rank_state,
            self.V.to(
                dtype=low_rank_state.dtype
            ),
        )
        
        input_term = torch.matmul(
            patch,
            self.W_input.transpose(0, 1).to(
                dtype=patch.dtype
            ),
        )

        candidate = torch.tanh(
            recurrent_term + input_term
        )

        return candidate

    @staticmethod
    def _straight_through_binary(
        probability: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Forward:
            hard binary mask

        Backward:
            gradient of probability
        """

        hard_mask = (
            probability >= threshold
        ).to(probability.dtype)

        return (
            hard_mask
            + probability
            - probability.detach()
        )

    def _training_step(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        patch: torch.Tensor,
        route_probability: torch.Tensor,
        force_keep: bool,
    ):
        if force_keep:
            route_mask = torch.ones_like(route_probability)
            effective_probability = torch.ones_like(route_probability)
        else:
            route_mask = self._straight_through_binary(
                route_probability,
                self.routing_threshold,
            )
            effective_probability = route_probability

        candidate = self._candidate_state(
            low_rank_state=q,
            patch=patch,
        )

        gate = torch.sigmoid(
            self.update_gate(patch)
        )

        proposed_h = (
            (1.0 - gate) * h
            + gate * candidate
        )

        proposed_h = self.state_norm(proposed_h)

        # Hard selection in forward, differentiable probability in backward.
        new_h = (
            (1.0 - route_mask) * h
            + route_mask * proposed_h
        )

        # q_t = h_t U
        new_q = torch.matmul(new_h, self.U)

        return (
            new_h,
            new_q,
            proposed_h,
            route_mask,
            effective_probability,
        )

    def _inference_step(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        patch: torch.Tensor,
        route_probability: torch.Tensor,
        force_keep: bool,
    ):
        batch_size = patch.shape[0]
    
        # =========================================================
        # Fast inference path for batch size = 1
        # PAMAP2 footprint / edge latency evaluation uses B=1.
        # Routing rule and state-update equation are unchanged.
        # =========================================================
        if batch_size == 1:
            if force_keep:
                should_update = True
            else:
                should_update = bool(
                    route_probability[0, 0].item()
                    >= self.routing_threshold
                )
    
            if should_update:
                candidate = self._candidate_state(
                    low_rank_state=q,
                    patch=patch,
                )
    
                gate = torch.sigmoid(
                    self.update_gate(patch)
                )
    
                new_h = (
                    (1.0 - gate) * h
                    + gate * candidate
                )
    
                new_h = self.state_norm(new_h)
                new_h = new_h.to(dtype=h.dtype)
    
                new_q = torch.matmul(
                    new_h,
                    self.U.to(dtype=new_h.dtype),
                )
                new_q = new_q.to(dtype=q.dtype)
    
                proposed_h = new_h
    
                hard_mask = torch.ones(
                    1,
                    1,
                    dtype=h.dtype,
                    device=h.device,
                )
    
                return (
                    new_h,
                    new_q,
                    proposed_h,
                    hard_mask,
                    hard_mask,
                )
    
            # Skip:
            # Reservoir candidate/gate/state update is not calculated.
            proposed_h = torch.zeros_like(h)
    
            hard_mask = torch.zeros(
                1,
                1,
                dtype=h.dtype,
                device=h.device,
            )
    
            return (
                h,
                q,
                proposed_h,
                hard_mask,
                hard_mask,
            )
    
        # =========================================================
        # Original general path for batch size > 1
        # =========================================================
        if force_keep:
            hard_mask = torch.ones(
                batch_size,
                1,
                dtype=patch.dtype,
                device=patch.device,
            )
        else:
            hard_mask = (
                route_probability >= self.routing_threshold
            ).to(dtype=patch.dtype)
    
        active_indices = torch.nonzero(
            hard_mask.squeeze(-1) > 0,
            as_tuple=False,
        ).squeeze(-1)
    
        proposed_h = torch.zeros_like(h)
    
        if active_indices.numel() > 0:
            active_h = h.index_select(
                0,
                active_indices,
            )
    
            active_q = q.index_select(
                0,
                active_indices,
            )
    
            active_patch = patch.index_select(
                0,
                active_indices,
            )
    
            active_candidate = self._candidate_state(
                low_rank_state=active_q,
                patch=active_patch,
            )
    
            active_gate = torch.sigmoid(
                self.update_gate(active_patch)
            )
    
            active_new_h = (
                (1.0 - active_gate) * active_h
                + active_gate * active_candidate
            )
    
            active_new_h = self.state_norm(
                active_new_h
            )
    
            active_new_h = active_new_h.to(
                dtype=h.dtype
            )
    
            h = h.clone()
            q = q.clone()
    
            h.index_copy_(
                0,
                active_indices,
                active_new_h,
            )
    
            active_new_q = torch.matmul(
                active_new_h,
                self.U.to(
                    dtype=active_new_h.dtype
                ),
            )
    
            active_new_q = active_new_q.to(
                dtype=q.dtype
            )
    
            q.index_copy_(
                0,
                active_indices,
                active_new_q,
            )
    
            proposed_h.index_copy_(
                0,
                active_indices,
                active_new_h.to(
                    dtype=proposed_h.dtype
                ),
            )
    
        hard_mask = hard_mask.to(
            dtype=h.dtype
        )
    
        return (
            h,
            q,
            proposed_h,
            hard_mask,
            hard_mask,
        )

    def forward(self, patches: torch.Tensor):
        """
        Args:
            patches:
                [B, T, D]

        Returns:
            final_state:
                [B, N]

            mean_selected_state:
                [B, N]

            routing_stats:
                dictionary for monitoring
        """

        batch_size, num_patches, _ = patches.shape

        h = torch.zeros(
            batch_size,
            self.reservoir_size,
            dtype=patches.dtype,
            device=patches.device,
        )

        # Low-rank representation q = hU.
        # Initial h is zero, so q is also zero.
        q = torch.zeros(
            batch_size,
            self.reservoir_rank,
            dtype=patches.dtype,
            device=patches.device,
        )

        selected_state_sum = torch.zeros_like(h)
        selected_count = torch.zeros(
            batch_size,
            1,
            dtype=patches.dtype,
            device=patches.device,
        )

        if self.collect_routing_stats or self.training:
            route_probabilities = []
            hard_masks = []
            innovation_distances = []
            innovation_cosines = []
        else:
            route_probabilities = None
            hard_masks = None
            innovation_distances = None
            innovation_cosines = None

        for patch_index in range(num_patches):
            patch = patches[:, patch_index, :]

            (
                route_probability,
                _,
                innovation_distance,
                innovation_cosine,
            ) = self.router(
                patch=patch,
                low_rank_state=q,
            )

            force_keep = (
                patch_index < self.minimum_keep_patches
            )

            if self.training:
                (
                    h,
                    q,
                    proposed_h,
                    route_mask,
                    effective_probability,
                ) = self._training_step(
                    h=h,
                    q=q,
                    patch=patch,
                    route_probability=route_probability,
                    force_keep=force_keep,
                )
            else:
                (
                    h,
                    q,
                    proposed_h,
                    route_mask,
                    effective_probability,
                ) = self._inference_step(
                    h=h,
                    q=q,
                    patch=patch,
                    route_probability=route_probability,
                    force_keep=force_keep,
                )

            selected_state_sum = (
                selected_state_sum
                + route_mask * proposed_h
            )

            selected_count = (
                selected_count
                + route_mask
            )

            if self.collect_routing_stats or self.training:
                route_probabilities.append(
                    effective_probability
                )
            
                hard_masks.append(
                    (
                        route_mask.detach()
                        >= 0.5
                    ).to(patches.dtype)
                )
            
                innovation_distances.append(
                    innovation_distance.detach()
                )
            
                innovation_cosines.append(
                    innovation_cosine.detach()
                )

        selected_count = selected_count.clamp_min(1.0)
        
        mean_selected_state = (
            selected_state_sum
            / selected_count
        )
        
        if self.collect_routing_stats or self.training:
            route_probability_tensor = torch.stack(
                route_probabilities,
                dim=1,
            ).squeeze(-1)
        
            hard_mask_tensor = torch.stack(
                hard_masks,
                dim=1,
            ).squeeze(-1)
        
            innovation_distance_tensor = torch.stack(
                innovation_distances,
                dim=1,
            ).squeeze(-1)
        
            innovation_cosine_tensor = torch.stack(
                innovation_cosines,
                dim=1,
            ).squeeze(-1)
        
            mean_route_probability = (
                route_probability_tensor.mean()
            )
        
            self.last_aux_loss = (
                mean_route_probability
                - self.target_keep_ratio
            ).pow(2)
        
            hard_keep_ratio = hard_mask_tensor.mean()
        
            self.last_routing_stats = {
                "route_probability_mean": float(
                    mean_route_probability.detach().cpu()
                ),
                "hard_keep_ratio": float(
                    hard_keep_ratio.detach().cpu()
                ),
                "selected_patches_mean": float(
                    hard_mask_tensor.sum(dim=1)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                ),
                "innovation_distance_mean": float(
                    innovation_distance_tensor.mean()
                    .detach()
                    .cpu()
                ),
                "innovation_cosine_mean": float(
                    innovation_cosine_tensor.mean()
                    .detach()
                    .cpu()
                ),
            }
        
        else:
            # Fast inference mode:
            # statistics are unnecessary for latency measurement.
            self.last_aux_loss = None
            self.last_routing_stats = {}

        return (
            h,
            q,
            mean_selected_state,
        )

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.last_aux_loss

    def get_routing_stats(self) -> Dict[str, float]:
        return dict(self.last_routing_stats)


class PatchReservoir(nn.Module):
    """
    Distilled State Innovation Routed LRGR classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 32,
        stride: Optional[int] = None,
        reservoir_size: int = 1000,
        reservoir_rank: int = 32,
        num_classes: int = 8,
        router_hidden_dim: int = 64,
        routing_threshold: float = 0.5,
        target_keep_ratio: float = 0.5,
        minimum_keep_patches: int = 1,
        collect_routing_stats: bool = True,
    ):
        super().__init__()

        if stride is None:
            stride = patch_size

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.reservoir_size = reservoir_size
        self.reservoir_rank = reservoir_rank
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
        )

        patch_dim = patch_size

        self.reservoir_network = (
            StateInnovationLowRankGatedReservoirNetwork(
                input_dim=patch_dim,
                reservoir_size=reservoir_size,
                reservoir_rank=reservoir_rank,
                router_hidden_dim=router_hidden_dim,
                routing_threshold=routing_threshold,
                target_keep_ratio=target_keep_ratio,
                minimum_keep_patches=minimum_keep_patches,
                collect_routing_stats=collect_routing_stats,
            )
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, patch_dim)
        )

        self.dist_token = nn.Parameter(
            torch.zeros(1, patch_dim)
        )

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.dist_token,
            std=0.02,
        )

        self.output_norm = nn.LayerNorm(
            reservoir_size
        )

        self.classification_head = nn.Linear(
            reservoir_size,
            num_classes,
        )

        self.distillation_head = nn.Linear(
            reservoir_size,
            num_classes,
        )

    def _token_state(
        self,
        low_rank_state: torch.Tensor,
        token: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = low_rank_state.shape[0]

        expanded_token = token.expand(
            batch_size,
            -1,
        )

        recurrent_term = torch.matmul(
            low_rank_state,
            self.reservoir_network.V,
        )

        token_term = torch.matmul(
            expanded_token,
            self.reservoir_network.W_input.transpose(0, 1),
        )

        return torch.tanh(
            recurrent_term + token_term
        )

    def forward(self, x: torch.Tensor):
        patches = self.patch_embed(x)

        (
            final_state,
            final_low_rank_state,
            mean_selected_state,
        ) = self.reservoir_network(patches)

        cls_state = self._token_state(
            low_rank_state=final_low_rank_state,
            token=self.cls_token,
        )

        dist_state = self._token_state(
            low_rank_state=final_low_rank_state,
            token=self.dist_token,
        )

        cls_state = self.output_norm(
            cls_state + mean_selected_state
        )

        dist_state = self.output_norm(
            dist_state + mean_selected_state
        )

        cls_output = self.classification_head(
            cls_state
        )

        dist_output = self.distillation_head(
            dist_state
        )

        if self.training:
            return cls_output, dist_output

        return (
            cls_output + dist_output
        ) / 2.0

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.reservoir_network.get_aux_loss()

    def get_routing_stats(self) -> Dict[str, float]:
        return self.reservoir_network.get_routing_stats()