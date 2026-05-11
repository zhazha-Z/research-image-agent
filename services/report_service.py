"""Markdown report generation for the MVP."""

from __future__ import annotations

from typing import Dict

from services.mock_analysis import DISCLAIMER
from services.rag_service import get_literature_references


def generate_markdown_report(analysis_result: Dict[str, object]) -> str:
    """Generate a structured Markdown experiment report from mock analysis output."""
    analysis_mode = analysis_result.get("analysis_mode", "mock 分析")
    metrics = analysis_result.get("metrics", [])
    visible_structures = analysis_result.get("visible_structures", [])
    abnormal_regions = analysis_result.get("abnormal_regions", [])
    references = get_literature_references()

    metric_lines = "\n".join(_format_metric_line(metric) for metric in metrics)
    structure_lines = "\n".join(f"- {item}" for item in visible_structures)
    abnormal_lines = "\n".join(
        f"- **{item['region']}**（置信度：{item['confidence']}）：{item['description']}"
        for item in abnormal_regions
    )
    limitations = analysis_result.get("limitations", [])
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    if not limitation_lines:
        limitation_lines = (
            "- 当前版本为 MVP mock 流程，不接入真实多模态大模型、分割模型或向量数据库。\n"
            "- 指标数值为演示数据，不能作为真实实验结论。\n"
            "- 后续可接入真实图像分割模型、显微图像专用提示词、文献向量检索、人工标注对照和报告导出功能。"
        )
    reference_lines = "\n".join(
        f"- **{item['method']}**：{item['summary']} 参考：{item['title']}。"
        for item in references
    )

    return f"""# 科研图像智能分析实验报告

> {DISCLAIMER}

## 一、实验目的

基于多模态大模型科研图像分析 Agent 的 mock 原型，对上传的细胞显微图像或实验图像进行结构识别、疑似异常区域解释、分割质量评估和文献方法辅助整理。

## 二、图像数据说明

- 文件名称：{analysis_result.get("file_name", "未知文件")}
- 图像尺寸：{analysis_result.get("image_size", "未知尺寸")}
- 图像质量判断：{analysis_result.get("image_quality", "mock 质量判断")}
- 分析模式：{analysis_mode}
- 数据类型：用户上传 JPG/PNG 科研图像。

## 三、图像分析结果

{analysis_result.get("overview", "")}

### 可见结构

{structure_lines}

### 分割结果说明

{analysis_result.get("segmentation_summary", "")}

## 四、指标解读

{metric_lines}

## 五、异常区域解释

{abnormal_lines}

## 六、相关文献方法参考

{reference_lines}

## 七、初步结论

本次分析给出了上传图像的初步结构化解读。相关结果需要结合实验设计、原始图像通道、人工标注与重复样本进一步确认，不能直接作为诊断或最终科研结论。

## 八、局限性与后续建议

{limitation_lines}

---

{DISCLAIMER}
"""


def _format_metric_line(metric: Dict[str, object]) -> str:
    value = metric.get("value")
    value_text = f"{value:.2f}" if isinstance(value, float) else "建议关注"
    return (
        f"- **{metric['name']} = {value_text}**："
        f"{metric['explanation']} {metric['interpretation']}"
    )
