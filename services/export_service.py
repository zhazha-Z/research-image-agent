"""Export helpers for reports, masks, CSV statistics, and JSON archives."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def save_mask_image(mask_image: Any, file_prefix: str) -> str:
    """Save a segmentation mask image as PNG and return the file path."""
    output_path = _build_output_path(file_prefix, "mask", "png")
    mask_image.save(output_path)
    return str(output_path)


def export_area_statistics_csv(segmentation_result: Dict[str, Any], file_prefix: str) -> str:
    """Export object areas and summary statistics to a CSV file."""
    output_path = _build_output_path(file_prefix, "area_statistics", "csv")
    stats = segmentation_result.get("area_statistics", {})
    area_list = stats.get("area_list", [])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["object_id", "area"])
        for index, area in enumerate(area_list, start=1):
            writer.writerow([index, area])
        writer.writerow([])
        writer.writerow(["summary", "value"])
        writer.writerow(["object_count", segmentation_result.get("object_count", 0)])
        writer.writerow(["mean_area", stats.get("mean_area", 0)])
        writer.writerow(["min_area", stats.get("min_area", 0)])
        writer.writerow(["max_area", stats.get("max_area", 0)])
        writer.writerow(["total_area", stats.get("total_area", 0)])

    return str(output_path)


def export_analysis_json(
    analysis_result: Dict[str, Any],
    chat_history: List[Dict[str, str]] | None = None,
    segmentation_result: Dict[str, Any] | None = None,
    file_prefix: str | None = None,
) -> str:
    """Export the full analysis archive to JSON and return the file path."""
    output_path = _build_output_path(file_prefix or "analysis", "analysis_result", "json")
    payload = {
        "analysis_result": _json_safe(analysis_result),
        "chat_history": _json_safe(chat_history or []),
        "segmentation_result": _json_safe(segmentation_result),
        "safety_note": "仅用于科研辅助分析，不构成医学诊断。",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def sanitize_filename(name: str) -> str:
    """Sanitize a user-provided file name for safe local output paths."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "analysis"


def _build_output_path(file_prefix: str, suffix: str, extension: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = sanitize_filename(file_prefix)
    return OUTPUTS_DIR / f"{safe_prefix}_{suffix}_{timestamp}.{extension}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "mask_image"}
    return str(value)
