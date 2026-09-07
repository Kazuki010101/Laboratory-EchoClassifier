import math

import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """Backward-compatible logit KD plus optional routing/content distillation."""

    def __init__(
        self,
        base_criterion,
        teacher_model,
        distillation_type,
        alpha,
        tau,
        route_weight=0.0,
        content_weight=0.0,
        teacher_layer=1,
        patch_size=16,
    ):
        super().__init__()

        supported_types = {
            "none",
            "soft",
            "hard",
            "soft2",
            "soft3",
            "soft4",
        }

        if distillation_type not in supported_types:
            raise ValueError(
                f"Unsupported distillation type: {distillation_type}"
            )

        self.base_criterion = base_criterion
        self.teacher_model = teacher_model

        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

        self.route_weight = route_weight
        self.content_weight = content_weight

        self.teacher_layer = teacher_layer
        self.patch_size = patch_size

        self.last = {
            "base_loss": None,
            "kd_loss": None,
            "route_loss": None,
            "content_loss": None,
            "t_entropy": None,
            "s_entropy": None,
            "agree": None,
            "alpha": alpha,
            "tau": tau,
        }

    @staticmethod
    def _fixed_projection(
        source_dim,
        target_dim,
        device,
        dtype,
    ):
        i = torch.arange(
            source_dim,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        j = torch.arange(
            target_dim,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        matrix = torch.cos(
            (i + 0.5)
            * (j + 1.0)
            * math.pi
            / source_dim
        )

        matrix = F.normalize(
            matrix,
            dim=0,
        )

        return matrix.to(dtype=dtype)

    @staticmethod
    def _unpack_student_output(student_output):
        if isinstance(student_output, torch.Tensor):
            return student_output, None, None

        if not isinstance(student_output, (tuple, list)):
            raise TypeError(
                "student_output must be a Tensor, tuple, or list"
            )

        if len(student_output) == 0:
            raise ValueError(
                "student_output must contain classification logits"
            )

        logits = student_output[0]

        logits_kd = (
            student_output[1]
            if len(student_output) > 1
            else None
        )

        aux = (
            student_output[2]
            if len(student_output) > 2
            else None
        )

        return logits, logits_kd, aux

    @staticmethod
    def _to_float(value):
        if value is None:
            return None

        return float(
            value.detach().item()
        )

    def forward(
        self,
        inputs,
        student_output,
        labels,
    ):
        logits, logits_kd, aux = (
            self._unpack_student_output(
                student_output
            )
        )

        base_loss = self.base_criterion(
            logits,
            labels,
        )

        kd_loss = None
        route_loss = None
        content_loss = None

        teacher_logits = None
        teacher_route = None
        teacher_evidence = None

        guided_method = (
            isinstance(aux, dict)
            and aux.get("method")
            in {
                "teacher_skip",
                "tgec",
            }
        )

        distillation_enabled = (
            self.distillation_type != "none"
        )

        guided_distillation_enabled = (
            distillation_enabled
            and guided_method
        )

        # --------------------------------------------------
        # Teacher-free training / downstream fine-tuning
        # --------------------------------------------------
        #
        # When distillation_type is "none", only the
        # classification loss is used.
        #
        # TG-Skip and TGEC may still return auxiliary
        # routing/content information, but SENvT is not
        # required during downstream fine-tuning.
        # --------------------------------------------------

        if not distillation_enabled:
            loss = base_loss

            with torch.no_grad():
                self.last.update(
                    {
                        "base_loss": self._to_float(
                            base_loss
                        ),
                        "kd_loss": None,
                        "route_loss": None,
                        "content_loss": None,
                        "t_entropy": None,
                        "s_entropy": None,
                        "agree": None,
                        "alpha": self.alpha,
                        "tau": self.tau,
                    }
                )

            return loss

        # --------------------------------------------------
        # Teacher-required distillation
        # --------------------------------------------------

        if self.teacher_model is None:
            raise ValueError(
                "Teacher model is required when "
                "distillation_type is not 'none'"
            )

        with torch.no_grad():
            if guided_distillation_enabled:
                (
                    teacher_logits,
                    teacher_route,
                    teacher_evidence,
                ) = (
                    self.teacher_model
                    .forward_with_temporal_evidence(
                        inputs,
                        self.teacher_layer,
                        self.patch_size,
                    )
                )
            else:
                teacher_logits = self.teacher_model(
                    inputs
                )

            if isinstance(
                teacher_logits,
                (tuple, list),
            ):
                teacher_logits = teacher_logits[0]

        # --------------------------------------------------
        # Logit distillation
        # --------------------------------------------------

        if logits_kd is None:
            raise ValueError(
                "KD requires a second student logit tensor"
            )

        if self.distillation_type == "soft":
            temperature = self.tau

            kd_loss = F.kl_div(
                F.log_softmax(
                    logits_kd / temperature,
                    dim=1,
                ),
                F.softmax(
                    teacher_logits / temperature,
                    dim=1,
                ),
                reduction="batchmean",
            ) * (temperature * temperature)

        elif self.distillation_type == "hard":
            kd_loss = F.cross_entropy(
                logits_kd,
                teacher_logits.argmax(dim=1),
            )

        else:
            raise NotImplementedError(
                "This package supports soft/hard KD "
                "for the new teacher-guided methods"
            )

        loss = (
            (1.0 - self.alpha) * base_loss
            + self.alpha * kd_loss
        )

        # --------------------------------------------------
        # Teacher-guided routing distillation
        # --------------------------------------------------

        if guided_distillation_enabled:
            if teacher_route is None:
                raise ValueError(
                    "Teacher route was not returned by "
                    "forward_with_temporal_evidence"
                )

            if "route_logits" not in aux:
                raise KeyError(
                    "Teacher-guided student output must "
                    "contain aux['route_logits']"
                )

            route_loss = F.kl_div(
                F.log_softmax(
                    aux["route_logits"],
                    dim=1,
                ),
                teacher_route,
                reduction="batchmean",
            )

            loss = (
                loss
                + self.route_weight * route_loss
            )

        # --------------------------------------------------
        # TGEC content distillation
        # --------------------------------------------------

        if (
            guided_distillation_enabled
            and aux.get("summary_token") is not None
            and self.content_weight > 0
        ):
            if teacher_evidence is None:
                raise ValueError(
                    "Teacher evidence was not returned by "
                    "forward_with_temporal_evidence"
                )

            if "omitted_mask" not in aux:
                raise KeyError(
                    "TGEC student output must contain "
                    "aux['omitted_mask']"
                )

            omitted_mask = (
                aux["omitted_mask"]
                .unsqueeze(-1)
                .to(
                    dtype=teacher_evidence.dtype,
                    device=teacher_evidence.device,
                )
            )

            omitted_count = (
                omitted_mask
                .sum(dim=1)
                .clamp_min(1.0)
            )

            target = (
                teacher_evidence
                * omitted_mask
            ).sum(dim=1) / omitted_count

            projection = self._fixed_projection(
                source_dim=target.shape[-1],
                target_dim=(
                    aux["summary_token"]
                    .shape[-1]
                ),
                device=target.device,
                dtype=target.dtype,
            )

            target = target @ projection

            content_loss = (
                1.0
                - F.cosine_similarity(
                    aux["summary_token"],
                    target,
                    dim=1,
                )
            ).mean()

            loss = (
                loss
                + self.content_weight
                * content_loss
            )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        with torch.no_grad():
            values = {
                "base_loss": self._to_float(
                    base_loss
                ),
                "kd_loss": self._to_float(
                    kd_loss
                ),
                "route_loss": self._to_float(
                    route_loss
                ),
                "content_loss": self._to_float(
                    content_loss
                ),
                "t_entropy": None,
                "s_entropy": None,
                "agree": None,
                "alpha": self.alpha,
                "tau": self.tau,
            }

            if (
                teacher_logits is not None
                and logits_kd is not None
            ):
                teacher_probability = F.softmax(
                    teacher_logits,
                    dim=1,
                )

                student_probability = F.softmax(
                    logits_kd,
                    dim=1,
                )

                teacher_entropy = -(
                    teacher_probability
                    * teacher_probability
                    .clamp_min(1e-12)
                    .log()
                ).sum(dim=1).mean()

                student_entropy = -(
                    student_probability
                    * student_probability
                    .clamp_min(1e-12)
                    .log()
                ).sum(dim=1).mean()

                agreement = (
                    teacher_logits.argmax(dim=1)
                    == logits_kd.argmax(dim=1)
                ).float().mean()

                values.update(
                    {
                        "t_entropy": self._to_float(
                            teacher_entropy
                        ),
                        "s_entropy": self._to_float(
                            student_entropy
                        ),
                        "agree": self._to_float(
                            agreement
                        ),
                    }
                )

            self.last.update(values)

        return loss