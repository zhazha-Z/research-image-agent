"""DashScope Qwen vision analysis service using the OpenAI-compatible API."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List

from services.mock_analysis import DISCLAIMER


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VISION_MODEL = "qwen-vl-plus"


QWEN_VISION_PROMPT = """
你是一个科研图像辅助分析助手。请基于用户上传的细胞显微图像或科研实验图像进行初步分析。

要求：
1. 只描述图像中可观察到的信息。
2. 不要做医学诊断。
3. 不确定的地方要明确说明。
4. 输出严格 JSON，不要输出 Markdown，不要输出代码块。
5. JSON 必须包含以下字段：
   - image_type：图像类型，例如 细胞显微图像、荧光显微图像、分割结果图、loss 曲线图、其他科研实验图像
   - image_type_confidence：图像类型判断置信度，0 到 1 之间的小数
   - image_summary：图像整体描述，字符串
   - analysis_focus：建议分析重点，数组
   - metric_explanation_mode：指标解释模式，只能从 segmentation、curve、fluorescence、phenotype、general_quality 中选择一个
   - visible_structures：图像中可见结构，数组
   - possible_abnormal_regions：疑似异常区域，数组
   - quality_notes：图像质量说明，数组
   - suggested_metrics：建议关注的分析指标，数组
   - limitations：局限性和不确定性，数组
   - safety_note：固定写“仅用于科研辅助分析，不构成医学诊断。”
""".strip()


def analyze_image_with_qwen(uploaded_file: Any) -> Dict[str, Any]:
    """Analyze an uploaded image with Qwen VL through DashScope."""
    _load_dotenv_if_available()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "未检测到 DASHSCOPE_API_KEY，请先配置环境变量。",
            "can_fallback_to_mock": True,
        }

    try:
        from openai import OpenAI

        image_data_url = _uploaded_file_to_data_url(uploaded_file)
        client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )
        completion = client.chat.completions.create(
            model=os.getenv("QWEN_VISION_MODEL", DEFAULT_QWEN_VISION_MODEL),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {
                            "type": "text",
                            "text": QWEN_VISION_PROMPT,
                        },
                    ],
                }
            ],
        )
        raw_text = completion.choices[0].message.content or ""
        return {
            "success": True,
            "data": _parse_qwen_json(raw_text),
            "raw_output": raw_text,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "can_fallback_to_mock": True,
        }


def qwen_result_to_app_result(
    qwen_data: Dict[str, Any],
    file_name: str,
    image_size: tuple[int, int],
) -> Dict[str, Any]:
    """Convert Qwen JSON output to the app's existing display/report shape."""
    width, height = image_size
    suggested_metrics = _ensure_list(qwen_data.get("suggested_metrics"))
    quality_notes = _ensure_list(qwen_data.get("quality_notes"))
    limitations = _ensure_list(qwen_data.get("limitations"))

    return {
        "analysis_mode": "千问 AI 分析",
        "file_name": file_name,
        "image_size": f"{width} x {height}px",
        "image_quality": "\n".join(quality_notes) if quality_notes else "千问未返回图像质量说明。",
        "overview": qwen_data.get("image_summary", "千问未返回图像概述。"),
        "visible_structures": _ensure_list(qwen_data.get("visible_structures")),
        "abnormal_regions": [
            {
                "region": f"疑似区域 {index}",
                "description": description,
                "confidence": "需人工复核",
            }
            for index, description in enumerate(
                _ensure_list(qwen_data.get("possible_abnormal_regions")),
                start=1,
            )
        ],
        "segmentation_summary": (
            "当前千问 AI 分析仅进行图像理解和科研辅助解读，未接入真实分割模型，"
            "因此不会伪造 Dice、IoU、Precision 或 Recall 数值。"
        ),
        "metrics": [
            {
                "name": metric,
                "value": None,
                "explanation": "千问建议关注该指标；具体数值需要结合分割模型输出和人工标注计算。",
                "interpretation": "建议作为后续定量分析或模型评估指标，不代表本次已经完成真实计算。",
            }
            for metric in suggested_metrics
        ],
        "quality_notes": quality_notes,
        "suggested_metrics": suggested_metrics,
        "limitations": limitations,
        "qwen_structured_result": qwen_data,
        "disclaimer": qwen_data.get("safety_note") or DISCLAIMER,
    }


def _parse_qwen_json(raw_text: str) -> Dict[str, Any]:
    cleaned_text = _strip_code_fence(raw_text.strip())
    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        extracted_text = _extract_json_object_text(cleaned_text)
        if not extracted_text:
            return _wrap_raw_text(raw_text)
        try:
            parsed = json.loads(extracted_text)
        except json.JSONDecodeError:
            return _wrap_raw_text(raw_text)

    if not isinstance(parsed, dict):
        return _wrap_raw_text(raw_text)

    parsed["image_summary"] = str(parsed.get("image_summary") or "千问未返回图像整体描述。")
    parsed["image_type"] = str(parsed.get("image_type") or "")
    parsed["image_type_confidence"] = _ensure_float(parsed.get("image_type_confidence"))
    parsed["analysis_focus"] = _ensure_list(parsed.get("analysis_focus"))
    parsed["metric_explanation_mode"] = str(parsed.get("metric_explanation_mode") or "")
    parsed["visible_structures"] = _ensure_list(parsed.get("visible_structures"))
    parsed["possible_abnormal_regions"] = _ensure_list(parsed.get("possible_abnormal_regions"))
    parsed["quality_notes"] = _ensure_list(parsed.get("quality_notes"))
    parsed["suggested_metrics"] = _ensure_list(parsed.get("suggested_metrics"))
    parsed["limitations"] = _ensure_list(parsed.get("limitations"))
    parsed["safety_note"] = DISCLAIMER
    return parsed


def _wrap_raw_text(raw_text: str) -> Dict[str, Any]:
    return {
        "image_type": "未明确识别",
        "image_type_confidence": 0.0,
        "image_summary": raw_text or "千问返回了空文本，无法解析为结构化 JSON。",
        "analysis_focus": [],
        "metric_explanation_mode": "general_quality",
        "visible_structures": [],
        "possible_abnormal_regions": [],
        "quality_notes": ["模型返回内容不是合法 JSON，已保留原始文本供人工查看。"],
        "suggested_metrics": [],
        "limitations": ["本次返回未能解析为合法 JSON，需要人工复核或重试。"],
        "safety_note": DISCLAIMER,
        "raw_text": raw_text,
    }


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _extract_json_object_text(text: str) -> str:
    start_index = text.find("{")
    if start_index < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]

    return ""


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _ensure_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 1.0))


def _uploaded_file_to_data_url(uploaded_file: Any) -> str:
    mime_type = getattr(uploaded_file, "type", None) or "image/png"
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
