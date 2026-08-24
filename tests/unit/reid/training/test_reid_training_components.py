import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from boxmot.engine.config import load_training_config
from boxmot.reid.backbones.anatomical_registry import (
    ANATOMICAL_TARGET_TYPES,
    DEFAULT_ANATOMICAL_TARGET_TYPE,
    V8_ANATOMICAL_TARGET_TYPE,
    get_anatomical_target_spec,
)
from boxmot.reid.backbones.families.csl_tinyvit.heads import (
    BranchSetAttention,
    ResidualMultiScaleQueryDecoder,
)
from boxmot.reid.backbones.head_registry import (
    MULTI_BRANCH_HEAD_TYPES,
    TRAIN_HEAD_TYPES,
    HeadImplementation,
    get_reid_head_spec,
)
from boxmot.reid.backbones.option_registry import (
    CSL_FEATURE_FUSION_MODES,
    normalize_selector,
    selector_choices,
)
from boxmot.reid.training.ablation import (
    CSL_TINYVIT_ADDONS,
    ActivationKind,
    AddonCategory,
    resolve_csl_tinyvit_ablation,
    validate_addon_registry,
)
from boxmot.reid.training.augmentations import validate_augmentation_config
from boxmot.reid.training.config import (
    AugmentationConfig,
    flatten_train_hparams,
)
from boxmot.reid.training.model_options import (
    REID_MODEL_OPTION_GROUPS,
    build_reid_model_kwargs,
)
from boxmot.reid.training.resume import contract_differences


def _ablation_options(**overrides):
    values = {
        spec.activation.field: (
            spec.activation.value
            if spec.activation.kind == ActivationKind.NOT_EQUALS
            else False
        )
        for spec in CSL_TINYVIT_ADDONS
    }
    for spec in CSL_TINYVIT_ADDONS:
        for setting in spec.settings:
            values.setdefault(setting, None)
    values.update(
        {
            "model_name": "csl_tinyvit_11m",
            "head_type": "standard",
            "head_pool": "avg",
            "head_parts": (1, 2, 4),
            "part_pooling": "stripes",
            "num_part_tokens": 4,
            "multiscale_channel_alpha": 0.5,
            "metric_feature": "raw_concat",
            "inference_feature": "norm_concat_bn",
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_head_registry_is_the_single_capability_source():
    assert "multiscale_channel2" in TRAIN_HEAD_TYPES
    assert "lmbn" not in TRAIN_HEAD_TYPES
    assert "gpc_lite" not in MULTI_BRANCH_HEAD_TYPES
    assert (
        get_reid_head_spec("multiscale_channel2").implementation
        == HeadImplementation.MULTI_BRANCH
    )
    assert get_reid_head_spec("multiscale_channel2").channel_control is True


def test_mode_registry_keeps_training_and_model_choices_aligned():
    assert (
        "global_final_parts_stage0_semantic_fine"
        in CSL_FEATURE_FUSION_MODES
    )
    assert "dpt_fpn" in selector_choices("feature_fusion")
    assert "dpt_fpn" not in CSL_FEATURE_FUSION_MODES
    assert normalize_selector("part_pooling", "STRIPES") == "stripes"
    with pytest.raises(ValueError, match="Unsupported feature_fusion"):
        normalize_selector("feature_fusion", "made_up")


def test_anatomical_registry_keeps_training_and_head_capabilities_aligned():
    assert DEFAULT_ANATOMICAL_TARGET_TYPE in ANATOMICAL_TARGET_TYPES
    v8 = get_anatomical_target_spec(V8_ANATOMICAL_TARGET_TYPE)
    assert v8.uses_ema_teacher is True
    assert v8.uses_semantic_teacher is False
    assert (
        get_anatomical_target_spec(
            "privileged_mask_pose_attention"
        ).uses_privileged_attention
        is True
    )
    with pytest.raises(ValueError, match="Unsupported anatomical target type"):
        get_anatomical_target_spec("unknown")


def test_saved_v8_hparams_use_resume_normalization_for_cfg(tmp_path):
    hparams = {
        "run": {"model_name": "csl_tinyvit_11m"},
        "model": {
            "head": {
                "anatomical_auxiliary": {
                    "token_dim": 128,
                    "multiscale": True,
                }
            }
        },
        "augmentation": {
            "anatomical_supervision": {
                "enabled": True,
                "teacher_momentum": 0.999,
            }
        },
        "resume": {"contract": {"version": 1}},
    }
    cfg_path = tmp_path / "hparams.json"
    cfg_path.write_text(json.dumps(hparams), encoding="utf-8")

    flattened = flatten_train_hparams(hparams)
    train_args = load_training_config(cfg_path)

    for values in (flattened, train_args):
        assert (
            values["anatomical_target_type"]
            == V8_ANATOMICAL_TARGET_TYPE
        )
        assert values["anatomical_foreground_weight"] == 0.0
        assert values["anatomical_semantic_part_weight"] == 0.0
    assert train_args["model"] == "csl_tinyvit_11m"
    assert "model_name" not in train_args


def test_v8_legacy_contract_is_semantically_equal_to_explicit_schema():
    legacy = {
        "version": 1,
        "model": {},
        "augmentation": {
            "anatomical_auxiliary": True,
            "anatomical_teacher_momentum": 0.999,
        },
    }
    explicit = {
        "version": 1,
        "model": {
            "anatomical_target_type": V8_ANATOMICAL_TARGET_TYPE,
        },
        "augmentation": {
            "anatomical_auxiliary": True,
            "anatomical_teacher_momentum": 0.999,
            "anatomical_foreground_weight": 0.0,
            "anatomical_semantic_part_weight": 0.0,
        },
    }

    assert contract_differences(legacy, explicit) == []
    explicit["augmentation"]["anatomical_foreground_weight"] = 0.1
    assert contract_differences(legacy, explicit) == [
        "augmentation.anatomical_foreground_weight: "
        "saved=0.0, requested=0.1"
    ]


def test_ablation_plan_groups_only_enabled_treatments():
    plan = resolve_csl_tinyvit_ablation(
        _ablation_options(
            head_type="multiscale_channel2",
            identity_registers=True,
            identity_register_count=4,
            identity_register_dim=128,
            identity_register_num_heads=4,
            identity_register_dropout=0.1,
            identity_register_gate_init=0.0,
            pav_mosaic=True,
            pav_metadata_dir="metadata",
            pav_mosaic_probability=0.25,
            csmm_loss_weight=0.2,
        )
    )

    assert plan.head.channel_control is True
    assert {
        addon.spec.name
        for addon in plan.by_category(AddonCategory.ARCHITECTURE)
    } == {"architecture.identity_registers"}
    assert {
        addon.spec.name
        for addon in plan.by_category(AddonCategory.AUGMENTATION)
    } == {"augmentation.pose_aligned_view_mosaic"}
    serialized = plan.to_dict()
    assert serialized["head"]["name"] == "multiscale_channel2"
    assert serialized["active"][0] == "head.multiscale_channel2"
    assert serialized["head"]["settings"]["head_parts"] == [1, 2, 4]


def test_width_first_ablation_records_its_compound_depth_allocation():
    plan = resolve_csl_tinyvit_ablation(
        _ablation_options(
            width_first_hierarchy=True,
            stage2_depth=5,
            stage3_depth=3,
        )
    )

    treatment = next(
        addon
        for addon in plan.addons
        if addon.spec.name == "architecture.width_first_hierarchy"
    )
    assert treatment.settings == {
        "width_first_hierarchy": True,
        "stage2_depth": 5,
        "stage3_depth": 3,
    }


def test_anatomical_ablation_records_teacher_and_loss_power():
    plan = resolve_csl_tinyvit_ablation(
        _ablation_options(
            anatomical_auxiliary=True,
            anatomical_target_type=V8_ANATOMICAL_TARGET_TYPE,
            anatomical_distill_weight=0.1,
            anatomical_attention_weight=0.1,
            anatomical_foreground_weight=0.0,
            anatomical_semantic_part_weight=0.0,
            anatomical_visibility_weight=0.05,
            anatomical_contrastive_weight=0.1,
            anatomical_descriptor_distill_weight=0.0,
            anatomical_branch_distill_weight=0.0,
            anatomical_branch_global_coefficient=0.2,
            anatomical_branch_coarse_coefficient=0.3,
            anatomical_branch_fine_coefficient=0.5,
            anatomical_pose_teacher_weight=0.03,
            anatomical_query_distill_weight=0.0,
            anatomical_query_relational_distill_weight=0.0,
            anatomical_query_diversity_weight=0.0,
            anatomical_query_diversity_margin=0.1,
            anatomical_part_triplet_weight=0.0,
            clean_student_consistency_weight=0.0,
            anatomical_local_scale_weight=0.6,
            anatomical_fine_scale_weight=0.4,
            anatomical_cross_scale_weight=0.05,
            anatomical_pose_only_reliability=0.0,
            anatomical_query_start_epoch=0,
            anatomical_query_ramp_end_epoch=0,
            anatomical_temperature=0.07,
            anatomical_teacher_momentum=0.999,
        )
    )
    teacher = plan.by_category(AddonCategory.SUPERVISION)[0]

    assert teacher.settings["anatomical_target_type"] == V8_ANATOMICAL_TARGET_TYPE
    assert teacher.settings["anatomical_pose_teacher_weight"] == 0.03
    assert teacher.settings["anatomical_foreground_weight"] == 0.0
    assert teacher.settings["anatomical_query_relational_distill_weight"] == 0.0
    assert teacher.settings["clean_student_consistency_weight"] == 0.0
    assert teacher.settings["anatomical_query_start_epoch"] == 0
    assert teacher.settings["anatomical_query_ramp_end_epoch"] == 0


def test_ablation_plan_allows_mcpt_with_training_only_anatomical_teacher():
    plan = resolve_csl_tinyvit_ablation(
        _ablation_options(
            model_name="csl_tinyvit_7m",
            mcpt_mode="shared_multiscale",
            anatomical_auxiliary=True,
        )
    )

    assert "architecture.mcpt" in plan.active_names
    assert "supervision.anatomical_teacher" in plan.active_names


def test_mobilenet_ablation_plan_records_promoted_v20_components():
    plan = resolve_csl_tinyvit_ablation(
        _ablation_options(
            model_name="mobilenetv4_conv_medium",
            head_type="standard",
            scale_balanced_branches=True,
            mcpt_mode="shared_multiscale",
            anatomical_auxiliary=True,
        )
    )

    assert plan.active_names == (
        "head.standard",
        "architecture.mcpt",
        "head.scale_balanced_branches",
        "supervision.anatomical_teacher",
    )


def test_addon_registry_is_internally_consistent():
    validate_addon_registry()


def test_ablation_plan_rejects_multiple_branch_communication_treatments():
    with pytest.raises(ValueError, match="branch_communication"):
        resolve_csl_tinyvit_ablation(
            _ablation_options(
                branch_set_attention=True,
                multiscale_query_decoder=True,
            )
        )


def test_model_kwarg_registry_has_unique_outputs_and_derived_switches():
    sources = {
        option.source or option.kwarg
        for group in REID_MODEL_OPTION_GROUPS
        for option in group.options
    }
    options = SimpleNamespace(**{source: 0 for source in sources})
    options.csmm_loss_weight = 0.2
    options.treeboost_loss_weight = 0.0
    options.anatomical_descriptor_distill_weight = 0.3
    options.anatomical_branch_distill_weight = 0.0

    kwargs = build_reid_model_kwargs(options)
    expected_count = sum(
        len(group.options) for group in REID_MODEL_OPTION_GROUPS
    )
    assert len(kwargs) == expected_count
    assert kwargs["return_cross_scale_features"] is True
    assert kwargs["return_treeboost_features"] is False
    assert kwargs["anatomical_descriptor_distill"] is True
    assert kwargs["anatomical_branch_distill"] is False
    assert kwargs["branch_metric"] == options.branch_aware_metric


def test_augmentation_validation_is_independent_of_trainer():
    config = replace(
        AugmentationConfig(),
        pav_mosaic=True,
        pav_metadata_dir="metadata",
        pav_mosaic_decay_start_epoch=90,
    )
    validate_augmentation_config(config, epochs=100)

    invalid = replace(config, pav_mosaic_probability=1.1)
    with pytest.raises(ValueError, match="pav_mosaic_probability"):
        validate_augmentation_config(invalid, epochs=100)


def test_zero_initialized_communication_outputs_still_receive_gradients():
    branches = torch.randn(2, 7, 8)

    branch_set = BranchSetAttention(
        input_dim=8,
        token_dim=8,
        num_heads=2,
    )
    branch_set(branches).square().sum().backward()
    assert torch.count_nonzero(branch_set.output_proj.weight.grad) > 0

    query_decoder = ResidualMultiScaleQueryDecoder(
        input_dim=8,
        token_dim=8,
        num_heads=2,
    )
    maps = (
        torch.randn(2, 8, 2, 1),
        torch.randn(2, 8, 4, 2),
        torch.randn(2, 8, 8, 4),
    )
    query_decoder(branches, maps).square().sum().backward()
    assert torch.count_nonzero(
        query_decoder.output_projection.weight.grad
    ) > 0
