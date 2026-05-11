"""OpenAI vision analysis service for uploaded research images."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

from services.mock_analysis import DISCLAIMER


DEFAULT_VISION_MODEL = "gpt-4.1-mini"


VISION_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_summary": {"type": "string"},
        "visible_structures": {"type": "array", "items": {"type": "string"}},
        "possible_abnormal_regions": {"type": "array", "items": {"type": "string"}},
        "quality_notes": {"type": "string"},
        "suggested_metrics": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "safety_note": {"type": "string"},
    },
    "required": [
        "image_summary",
        "visible_structures",
        "possible_abnormal_regions",
        "quality_notes",
        "suggested_metrics",
        "limitations",
        "safety_note",
    ],
}


def analyze_image_with_openai(image_file: Any) -> Dict[str, Any]:
    """Analyze an uploaded image with OpenAI and return structured JSON or an error."""
    _load_dotenv_if_available()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "未检测到环境变量 OPENAI_API_KEY。请配置后再使用真实 AI 分析，或回退到 mock 分析。",
            "can_fallback_to_mock": True,
        }

    try:
        from openai import OpenAI

        data_url = _image_file_to_data_url(image_file)
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_VISION_MODEL", DEFAULT_VISION_MODEL),
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "你是科研图像辅助分析助手。只进行科研辅助解读，不做医学诊断，"
                                "不声称识别结果一定正确。必须输出符合给定 JSON Schema 的中文 JSON。"
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "请分析这张科研图像。重点描述整体内容、可见结构、疑似异常区域、"
                                "图像质量、建议关注的分析指标和不确定性。不要给出医学诊断。"
                            ),
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "research_image_analysis",
                    "schema": VISION_ANALYSIS_SCHEMA,
                    "strict": True,
                }
            },
        )
        return {
            "success": True,
            "data": json.loads(response.output_text),
            "raw_output": response.output_text,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"真实 AI 分析失败：{exc}",
            "can_fallback_to_mock": True,
        }


def openai_result_to_app_result(
    openai_data: Dict[str, Any],
    file_name: str,
    image_size: tuple[int, int],
) -> Dict[str, Any]:
    """Convert OpenAI JSON output to the app's existing display/report shape."""
    width, height = image_size
    suggested_metrics = openai_data.get("suggested_metrics", [])
    limitations = openai_data.get("limitations", [])

    return {
        "analysis_mode": "真实 AI 分析",
        "file_name": file_name,
        "image_size": f"{width} x {height}px",
        "image_quality": openai_data.get("quality_notes", "AI 未返回图像质量说明。"),
        "overview": openai_data.get("image_summary", "AI 未返回图像概述。"),
        "visible_structures": openai_data.get("visible_structures", []),
        "abnormal_regions": [
            {
                "region": f"疑似区域 {index}",
                "description": description,
                "confidence": "需人工复核",
            }
            for index, description in enumerate(
                openai_data.get("possible_abnormal_regions", []),
                start=1,
            )
        ],
        "segmentation_summary": (
            "当前真实 AI 分析仅进行图像语义解读，未接入真实分割模型，"
            "因此不会伪造 Dice、IoU、Precision 或 Recall 数值。"
        ),
        "metrics": [
            {
                "name": metric,
                "value": None,
                "explanation": "真实 AI 建议关注该指标；具体数值需要结合分割模型输出和人工标注计算。",
                "interpretation": "建议作为后续定量分析或模型评估指标，不代表本次已经完成真实计算。",
            }
            for metric in suggested_metrics
        ],
        "limitations": limitations,
        "openai_structured_result": openai_data,
        "disclaimer": openai_data.get("safety_note") or DISCLAIMER,
    }


def _image_file_to_data_url(image_file: Any) -> str:
    mime_type = getattr(image_file, "type", None) or "image/png"
    encoded = base64.b64encode(image_file.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
