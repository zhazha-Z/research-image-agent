"""Basic threshold segmentation and area statistics for microscopy images."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image
from skimage import color, filters, measure, morphology


SEGMENTATION_LIMITATIONS = (
    "该方法为基础阈值分割，仅适合对比度明显的图像，不适合作为最终科研结论。"
    "当前分割为基础阈值方法，需要人工标注或专业模型进一步验证，不能作为最终实验结论。"
)


def run_basic_segmentation(uploaded_file: Any) -> Dict[str, Any]:
    """Run Otsu thresholding, connected components, and area statistics."""
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        image_array = np.asarray(image)
        gray_image = color.rgb2gray(image_array)

        threshold = filters.threshold_otsu(gray_image)
        binary_mask = gray_image > threshold
        min_size = _estimate_min_object_size(binary_mask)
        cleaned_mask = morphology.remove_small_objects(binary_mask, max_size=min_size)
        cleaned_mask = morphology.remove_small_holes(cleaned_mask, max_size=min_size)

        labeled_mask = measure.label(cleaned_mask)
        regions = measure.regionprops(labeled_mask)
        area_list = [int(region.area) for region in regions]
        mask_image = Image.fromarray(cleaned_mask.astype(np.uint8) * 255, mode="L")

        return {
            "success": True,
            "mask_image": mask_image,
            "object_count": len(area_list),
            "area_statistics": {
                "area_list": area_list,
                "mean_area": float(np.mean(area_list)) if area_list else 0.0,
                "min_area": int(np.min(area_list)) if area_list else 0,
                "max_area": int(np.max(area_list)) if area_list else 0,
                "total_area": int(np.sum(area_list)) if area_list else 0,
            },
            "segmentation_method": "Otsu threshold + connected components",
            "limitations": SEGMENTATION_LIMITATIONS,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


def _estimate_min_object_size(binary_mask: np.ndarray) -> int:
    image_area = int(binary_mask.shape[0] * binary_mask.shape[1])
    return max(16, image_area // 2000)
