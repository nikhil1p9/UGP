from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path

from .analysis import VideoAnalyzer
from .config import AnalyzerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze driving video(s) and export structured JSON.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input: single file XOR batch folder (mutually exclusive, one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", metavar="VIDEO",
        help="Path to a single .mp4 input video.",
    )
    input_group.add_argument(
        "--batch-input", metavar="FOLDER",
        help="Folder containing .mp4 files. All will be processed in order.",
    )

    parser.add_argument(
        "--output", default=None,
        help=(
            "Single mode : path to the output .json file\n"
            "             (default: <video_stem>_analysis.json next to the input)\n"
            "Batch mode  : output folder for per-video .json files\n"
            "             (default: <batch-input>/results/)"
        ),
    )
    parser.add_argument("--interval", type=int, default=5,
                        help="Output JSON results for every N seconds (default: 5).")
    parser.add_argument("--sample-every",    type=int,   default=3,
                        help="Sample every Nth frame (default: 3).")
    parser.add_argument("--min-confidence",  type=float, default=0.35,
                        help="Detection confidence threshold (default: 0.35).")
    parser.add_argument("--detector-model",  default="yolov8n.pt",
                        help="Ultralytics YOLO model file (default: yolov8n.pt).")
    parser.add_argument("--lane-backend",    choices=["heuristic", "clrnet"],
                        default="heuristic")
    parser.add_argument("--enable-vlm",      action="store_true",
                        help="Refine result with OpenAI VLM (requires OPENAI_API_KEY).")
    parser.add_argument("--max-vlm-frames",  type=int,   default=6,
                        help="Max key frames sent to the VLM (default: 6).")
    
    return parser


def _make_config(input_path: Path, output_path: Path, args: argparse.Namespace) -> AnalyzerConfig:
    return AnalyzerConfig(
        input_path=input_path,
        output_path=output_path,
        interval_seconds=args.interval,
        sample_every_n_frames=args.sample_every,
        min_detection_confidence=args.min_confidence,
        detector_model=args.detector_model,
        lane_backend=args.lane_backend,
        enable_vlm=args.enable_vlm,
        max_vlm_frames=args.max_vlm_frames,
    )


def _run_single(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / (input_path.stem + "_analysis.json")
    )
    config = _make_config(input_path, output_path, args)
    results = VideoAnalyzer(config).analyze()
    
    # We now dump a list of results
    print(json.dumps([r.model_dump() for r in results], indent=2))


def _run_batch(args: argparse.Namespace) -> None:
    input_dir  = Path(args.batch_input)
    output_dir = Path(args.output) if args.output else input_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Batch: {len(videos)} video(s)  →  {output_dir}", file=sys.stderr)
    for video_path in videos:
        out_path = output_dir / (video_path.stem + ".json")
        print(f"  Processing {video_path.name} ...", end=" ", flush=True, file=sys.stderr)
        try:
            config = _make_config(video_path, out_path, args)
            VideoAnalyzer(config).analyze()
            print(f"→ {out_path.name}", file=sys.stderr)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_input:
        _run_batch(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
