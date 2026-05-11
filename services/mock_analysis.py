"""Mock image analysis service for the Streamlit MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DISCLAIMER = "仅用于科研辅助分析，不构成医学诊断。"


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    explanation: str
    interpretation: str


def run_mock_analysis(file_name: str, image_size: tuple[int, int]) -> Dict[str, object]:
    """Return deterministic mock analysis content for an uploaded research image."""
    width, height = image_size
    image_area_level = "高分辨率" if width * height >= 1_000_000 else "常规分辨率"

    metrics = [
        MetricResult(
            name="Dice",
            value=0.86,
            explanation="Dice 衡量预测分割区域与参考标注区域的重叠程度，越接近 1 表示重叠越充分。",
            interpretation="当前 Dice 为 0.86，说明 mock 分割结果与人工参考区域整体重合度较高。",
        ),
        MetricResult(
            name="IoU",
            value=0.76,
            explanation="IoU 表示交集面积占并集面积的比例，比 Dice 更严格，越接近 1 越好。",
            interpretation="当前 IoU 为 0.76，提示目标轮廓大体可靠，但边界仍可能存在少量偏差。",
        ),
        MetricResult(
            name="Precision",
            value=0.89,
            explanation="Precision 表示模型预测为目标的区域中，有多少确实属于目标区域。",
            interpretation="当前 Precision 为 0.89，说明误分进来的背景区域相对较少。",
        ),
        MetricResult(
            name="Recall",
            value=0.82,
            explanation="Recall 表示真实目标区域中，有多少被模型成功识别出来。",
            interpretation="当前 Recall 为 0.82，说明仍有少量目标区域可能被漏检。",
        ),
    ]

    visible_structures = [
        "图像中可见多个边界相对清晰的疑似细胞或组织结构。",
        "局部区域存在亮度增强，可能对应染色较强、荧光信号聚集或样本厚度差异。",
        "背景噪声处于中等水平，部分边缘区域可能影响自动分割稳定性。",
    ]

    abnormal_regions = [
        {
            "region": "右上象限局部高亮区域",
            "description": "该区域在 mock 分析中被标记为疑似异常信号聚集，可能与细胞密度升高、荧光过表达或成像伪影有关。",
            "confidence": "中等",
        },
        {
            "region": "中心偏左边界模糊区域",
            "description": "该区域边缘对比度较低，分割边界可能不稳定，建议结合原始实验记录或人工标注复核。",
            "confidence": "中等偏低",
        },
    ]

    return {
        "file_name": file_name,
        "image_size": f"{width} x {height}px",
        "image_quality": image_area_level,
        "overview": (
            f"已对上传图像 `{file_name}` 完成 mock 多模态分析。图像尺寸为 "
            f"{width} x {height}px，属于{image_area_level}科研图像。整体来看，图像包含可分辨的目标结构、"
            "局部信号增强区域以及少量边界模糊区域，适合用于演示细胞显微图像或实验图像的智能分析流程。"
        ),
        "visible_structures": visible_structures,
        "abnormal_regions": abnormal_regions,
        "segmentation_summary": (
            "mock 分割模型识别出主要目标区域，并给出边界轮廓说明。整体分割效果较好，目标主体基本被覆盖；"
            "但在亮度突变和边缘模糊处可能出现轻微过分割或漏分割。建议在真实项目中加入人工标注对照、"
            "多模型结果比对和不确定性可视化。"
        ),
        "metrics": [metric.__dict__ for metric in metrics],
        "disclaimer": DISCLAIMER,
    }


def metric_help_text() -> List[str]:
    """Return concise metric guidance for the UI."""
    return [
        "Dice 更关注预测区域和真实区域的整体重叠，常用于医学图像与细胞分割。",
        "IoU 更严格，分母是预测与真实区域的并集，因此通常低于 Dice。",
        "Precision 高表示误检少，适合判断背景是否被错误分进目标。",
        "Recall 高表示漏检少，适合判断真实目标是否被充分找出。",
    ]
