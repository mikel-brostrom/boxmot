"""Command adapter for offline HP-GRD teacher-signal extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="Registered ReID teacher checkpoint")
    parser.add_argument("--model-name", help="Teacher registry name when it cannot be inferred")
    parser.add_argument(
        "--part-names",
        nargs="+",
        help="Ordered semantic part names; canonical six-part anatomy is the default",
    )
    parser.add_argument("--dataset-index", type=Path, required=True, help="Stable JSON/JSONL sample index")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for relative image paths")
    mask_source = parser.add_mutually_exclusive_group(required=True)
    mask_source.add_argument("--anatomical-metadata", type=Path, help="Pose/anatomical metadata directory")
    mask_source.add_argument("--part-mask-input", type=Path, help="Tensor bundle with [N,P,H,W] masks")
    parser.add_argument("--person-mask-dir", type=Path, help="Optional external person parser masks")
    parser.add_argument("--output", type=Path, required=True, help="Output consumed by privileged_cache build")
    parser.add_argument("--img-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--preprocess", choices=("resize", "resize_pad"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--descriptor-key", help="Mapping path for a non-standard teacher output")
    parser.add_argument("--include-leave-part-out", action="store_true")
    parser.add_argument("--global-confidence-from-parts", action="store_true")
    parser.add_argument("--fill-value", type=float, default=0.0, help="Fill in normalized RGB space")
    parser.add_argument("--max-intervention-batch", type=int)
    parser.add_argument("--no-normalize", action="store_true", help="Store raw teacher descriptors")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone registered-teacher extractor."""

    args = _build_parser().parse_args(argv)

    import torch

    from boxmot.reid.training.teacher_extraction import (
        TeacherExtractionConfig,
        run_teacher_extraction,
    )

    result = run_teacher_extraction(
        TeacherExtractionConfig(
            teacher_checkpoint=args.teacher,
            model_name=args.model_name,
            part_names=None if args.part_names is None else tuple(args.part_names),
            dataset_index=args.dataset_index,
            image_root=args.image_root,
            anatomical_metadata=args.anatomical_metadata,
            person_mask_dir=args.person_mask_dir,
            part_mask_input=args.part_mask_input,
            output=args.output,
            img_size=None if args.img_size is None else tuple(args.img_size),
            preprocess=args.preprocess,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            amp=not args.no_amp,
            descriptor_key=args.descriptor_key,
            include_leave_part_out=args.include_leave_part_out,
            global_confidence_from_parts=args.global_confidence_from_parts,
            fill_value=args.fill_value,
            max_intervention_batch=args.max_intervention_batch,
            normalize_descriptors=not args.no_normalize,
            storage_dtype=getattr(torch, args.storage_dtype),
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result.summary(), sort_keys=True))
    return 0


__all__ = ("main",)


if __name__ == "__main__":
    raise SystemExit(main())
