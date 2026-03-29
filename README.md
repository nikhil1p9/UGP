# Driving Video To JSON

This project analyzes a driving video and exports structured JSON with:

- actors and counts
- per-actor actions
- scene summary
- vehicle types
- relative speed buckets: `stopped`, `slow`, `moderate`, `fast`
- estimated number of lanes
- event label: `normal`, `near_miss`, `collision`

The pipeline is inspired by SeeUnsafe-style traffic video reasoning, with a practical local implementation:

- object detection and lightweight tracking
- heuristic lane estimation by default
- optional VLM refinement for scene and event context
- JSON export for downstream evaluation

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

**Single video** – output defaults to `<video_stem>_analysis.json` next to the input:

```powershell
python -m drive_text.cli --input path\to\video.mp4
# or specify the output explicitly:
python -m drive_text.cli --input path\to\video.mp4 --output outputs\result.json
```

**Batch folder** – processes every `.mp4` in a folder; results go to `<folder>/results/` by default:

```powershell
python -m drive_text.cli --batch-input path\to\folder
# or specify a custom output folder:
python -m drive_text.cli --batch-input path\to\folder --output path\to\results
```

**Optional VLM refinement** with OpenAI (adds scene/event reasoning on top of CV analysis):

```powershell
$env:OPENAI_API_KEY="your_key"
python -m drive_text.cli --input path\to\video.mp4 --enable-vlm
```

## Output Schema

The generated JSON includes:

- `video_metadata`
- `scene`
- `lanes`
- `actors`
- `event`
- `diagnostics`

## Notes

- Relative speed is estimated from tracked object motion in image space. It is not absolute physical speed.
- Lane count uses a heuristic backend by default. You can extend `drive_text.lane` to call CLRNet if you install it separately.
- VLM refinement is optional. Without an API key, the pipeline still returns JSON from local CV components.
