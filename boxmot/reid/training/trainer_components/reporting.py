"""Training metrics, plots, and output-directory management."""

from __future__ import annotations

from pathlib import Path
from typing import List

from boxmot.reid.training.trainer_components.types import (
    TrainMetrics,
    ValMetrics,
)
from boxmot.utils import logger as LOGGER


class _ReportingMixin:
    def _save_metrics(
        self,
        save_dir: Path,
        history: List[TrainMetrics],
        val_history: List[ValMetrics],
        best_epoch: int,
        best_mAP: float,
        best_rank1: float,
        average_epoch_time_s: float = 0.0,
        average_forward_time_s: float = 0.0,
        average_eval_time_s: float = 0.0,
        total_end_to_end_time_s: float = 0.0,
    ):
        """Persist full training & validation history to metrics.json."""
        # Group val entries by epoch
        from collections import OrderedDict

        val_by_epoch: OrderedDict[int, dict] = OrderedDict()
        for v in val_history:
            if v.epoch not in val_by_epoch:
                val_by_epoch[v.epoch] = {"epoch": v.epoch}
            val_by_epoch[v.epoch][v.dataset] = {
                "mAP": round(v.mAP, 4),
                "rank1": round(v.rank1, 4),
                "rank5": round(v.rank5, 4),
                "rank10": round(v.rank10, 4),
            }
            if v.mcpt_disabled_mAP is not None:
                val_by_epoch[v.epoch][v.dataset].update(
                    mcpt_disabled_mAP=round(v.mcpt_disabled_mAP, 4),
                    mcpt_disabled_rank1=round(v.mcpt_disabled_rank1 or 0.0, 4),
                    mcpt_mAP_delta=round(v.mAP - v.mcpt_disabled_mAP, 4),
                )

        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_mAP": round(best_mAP, 4),
            "best_rank1": round(best_rank1, 4),
            "average_epoch_time_s": round(average_epoch_time_s, 4),
            "average_forward_time_s": round(average_forward_time_s, 4),
            "average_eval_time_s": round(average_eval_time_s, 4),
            "total_end_to_end_time_s": round(total_end_to_end_time_s, 4),
            "train": [
                {
                    "epoch": m.epoch,
                    "loss": round(m.loss, 5),
                    "id_loss": round(m.id_loss, 5),
                    "triplet_loss": round(m.triplet_loss, 5),
                    "adasp_loss": round(m.adasp_loss, 5),
                    "part_relation_loss": round(m.part_relation_loss, 5),
                    "part_to_global_loss": round(m.part_to_global_loss, 5),
                    "jpm_id_loss": round(m.jpm_id_loss, 5),
                    "jpm_metric_loss": round(m.jpm_metric_loss, 5),
                    "multilevel_suppression_loss": round(
                        m.multilevel_suppression_loss,
                        5,
                    ),
                    "multilevel_suppression_weight": round(
                        m.multilevel_suppression_weight,
                        7,
                    ),
                    "multilevel_suppression_effective_ratio": round(
                        m.multilevel_suppression_effective_ratio,
                        7,
                    ),
                    "multilevel_suppression_coarse_erased_fraction": round(
                        m.multilevel_suppression_coarse_erased_fraction,
                        7,
                    ),
                    "multilevel_suppression_fine_erased_fraction": round(
                        m.multilevel_suppression_fine_erased_fraction,
                        7,
                    ),
                    "multilevel_suppression_global_cam_active_fraction": round(
                        m.multilevel_suppression_global_cam_active_fraction,
                        7,
                    ),
                    "multilevel_suppression_coarse_cam_active_fraction": round(
                        m.multilevel_suppression_coarse_cam_active_fraction,
                        7,
                    ),
                    "center_loss": round(m.center_loss, 5),
                    "csmm_loss": round(m.csmm_loss, 5),
                    "treeboost_loss": round(m.treeboost_loss, 5),
                    "global_ap_loss": round(m.global_ap_loss, 5),
                    "hpgrd_loss": round(m.hpgrd_loss, 5),
                    "hpgrd_global_loss": round(m.hpgrd_global_loss, 5),
                    "hpgrd_part_loss": round(m.hpgrd_part_loss, 5),
                    "hpgrd_background_loss": round(m.hpgrd_background_loss, 5),
                    "hpgrd_part_drop_loss": round(m.hpgrd_part_drop_loss, 5),
                    "hpgrd_gradient_scale": round(m.hpgrd_gradient_scale, 7),
                    "late_interaction_loss": round(m.late_interaction_loss, 5),
                    "late_interaction_distill_loss": round(m.late_interaction_distill_loss, 5),
                    "pav_consistency_loss": round(m.pav_consistency_loss, 5),
                    "clean_student_consistency_loss": round(
                        m.clean_student_consistency_loss,
                        5,
                    ),
                    "anatomical_loss": round(m.anatomical_loss, 5),
                    "anatomical_distill_loss": round(
                        m.anatomical_distill_loss,
                        5,
                    ),
                    "anatomical_attention_loss": round(
                        m.anatomical_attention_loss,
                        5,
                    ),
                    "anatomical_visibility_loss": round(
                        m.anatomical_visibility_loss,
                        5,
                    ),
                    "anatomical_contrastive_loss": round(
                        m.anatomical_contrastive_loss,
                        5,
                    ),
                    "anatomical_descriptor_distill_loss": round(
                        m.anatomical_descriptor_distill_loss,
                        5,
                    ),
                    "anatomical_branch_distill_loss": round(
                        m.anatomical_branch_distill_loss,
                        5,
                    ),
                    "anatomical_branch_global_loss": round(
                        m.anatomical_branch_global_loss,
                        5,
                    ),
                    "anatomical_branch_coarse_loss": round(
                        m.anatomical_branch_coarse_loss,
                        5,
                    ),
                    "anatomical_branch_fine_loss": round(
                        m.anatomical_branch_fine_loss,
                        5,
                    ),
                    "anatomical_pose_teacher_loss": round(
                        m.anatomical_pose_teacher_loss,
                        5,
                    ),
                    "anatomical_semantic_foreground_loss": round(
                        m.anatomical_semantic_foreground_loss,
                        5,
                    ),
                    "anatomical_semantic_part_loss": round(
                        m.anatomical_semantic_part_loss,
                        5,
                    ),
                    "anatomical_query_distill_loss": round(
                        m.anatomical_query_distill_loss,
                        5,
                    ),
                    "anatomical_query_relational_distill_loss": round(
                        m.anatomical_query_relational_distill_loss,
                        5,
                    ),
                    "anatomical_query_diversity_loss": round(
                        m.anatomical_query_diversity_loss,
                        5,
                    ),
                    "anatomical_part_triplet_loss": round(
                        m.anatomical_part_triplet_loss,
                        5,
                    ),
                    "anatomical_accessory_valid_fraction": round(
                        m.anatomical_accessory_valid_fraction,
                        5,
                    ),
                    "identity_register_diversity_loss": round(
                        m.identity_register_diversity_loss,
                        5,
                    ),
                    "mcpt_loss": round(m.mcpt_loss, 7),
                    "mcpt_smoothness": round(m.mcpt_smoothness, 7),
                    "mcpt_identity": round(m.mcpt_identity, 7),
                    "mcpt_mean_abs_displacement": round(m.mcpt_mean_abs_displacement, 7),
                    "mcpt_boundary_1": round(m.mcpt_boundary_1, 7),
                    "mcpt_boundary_2": round(m.mcpt_boundary_2, 7),
                    "mcpt_boundary_3": round(m.mcpt_boundary_3, 7),
                    "mcpt_boundary_std": round(m.mcpt_boundary_std, 7),
                    "mcpt_cap_fraction": round(m.mcpt_cap_fraction, 7),
                    "mcpt_local_gate": round(m.mcpt_local_gate, 7),
                    "mcpt_fine_gate": round(m.mcpt_fine_gate, 7),
                    "anatomical_local_scale_loss": round(
                        m.anatomical_local_scale_loss,
                        5,
                    ),
                    "anatomical_fine_scale_loss": round(
                        m.anatomical_fine_scale_loss,
                        5,
                    ),
                    "anatomical_cross_scale_loss": round(
                        m.anatomical_cross_scale_loss,
                        5,
                    ),
                    "anatomical_valid_part_fraction": round(
                        m.anatomical_valid_part_fraction,
                        5,
                    ),
                    "anatomical_cross_camera_anchor_fraction": round(
                        m.anatomical_cross_camera_anchor_fraction,
                        5,
                    ),
                    "lr": round(m.lr, 8),
                    "backbone_lr": round(m.backbone_lr, 8),
                    "head_lr": round(m.head_lr, 8),
                }
                for m in history
            ],
            "val": list(val_by_epoch.values()),
        }
        path = save_dir / "metrics.json"
        self._write_json_atomic(path, data)
        LOGGER.info(f"Saved training metrics to {path}")

    def _save_training_plots(
        self,
        save_dir: Path,
        history: List[TrainMetrics],
        val_history: List[ValMetrics],
    ) -> None:
        """Plot training losses and primary validation metrics after training."""
        if not history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            LOGGER.warning(f"Could not generate training plots: {exc}")
            return

        train_epochs = [m.epoch for m in history]
        primary_dataset = self.dataset_name.split(",")[0].strip().lower()
        val_by_epoch: dict[int, ValMetrics] = {}
        for val in val_history:
            val_ds = val.dataset.strip().lower()
            if val_ds == primary_dataset and val.epoch not in val_by_epoch:
                val_by_epoch[val.epoch] = val

        if not val_by_epoch:
            for val in val_history:
                val_by_epoch.setdefault(val.epoch, val)

        val_epochs = sorted(val_by_epoch)
        mAP = [val_by_epoch[epoch].mAP for epoch in val_epochs]
        rank1 = [val_by_epoch[epoch].rank1 for epoch in val_epochs]
        rank5 = [val_by_epoch[epoch].rank5 for epoch in val_epochs]
        rank10 = [val_by_epoch[epoch].rank10 for epoch in val_epochs]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=140)

        loss_ax = axes[0]
        loss_ax.plot(train_epochs, [m.loss for m in history], label="loss", linewidth=2)
        loss_ax.plot(train_epochs, [m.id_loss for m in history], label="id_loss")
        loss_ax.plot(train_epochs, [m.triplet_loss for m in history], label="triplet_loss")
        loss_ax.plot(train_epochs, [m.center_loss for m in history], label="center_loss")
        if any(m.multilevel_suppression_loss > 0 for m in history):
            loss_ax.plot(
                train_epochs,
                [m.multilevel_suppression_loss for m in history],
                label="multilevel_suppression_loss",
            )
        loss_ax.set_title("Training Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, alpha=0.3)
        loss_ax.legend()

        metrics_ax = axes[1]
        if val_epochs:
            metrics_ax.plot(val_epochs, mAP, label="mAP", linewidth=2)
            metrics_ax.plot(val_epochs, rank1, label="Rank-1")
            metrics_ax.plot(val_epochs, rank5, label="Rank-5")
            metrics_ax.plot(val_epochs, rank10, label="Rank-10")
            metrics_ax.set_ylim(0.0, 1.0)
        metrics_ax.set_title(f"Validation Metrics ({primary_dataset})")
        metrics_ax.set_xlabel("Epoch")
        metrics_ax.set_ylabel("Score")
        metrics_ax.grid(True, alpha=0.3)
        if val_epochs:
            metrics_ax.legend()

        fig.tight_layout()
        path = save_dir / "training_curves.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info(f"Saved training curves to {path}")

    def _make_save_dir(self) -> Path:
        base = self.project / self.name
        if base.exists():
            idx = 1
            while (self.project / f"{self.name}_{idx}").exists():
                idx += 1
            base = self.project / f"{self.name}_{idx}"
        base.mkdir(parents=True, exist_ok=True)
        return base
