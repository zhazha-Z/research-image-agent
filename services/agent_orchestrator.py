"""Agent-style orchestration for uploaded research-image analysis."""

from __future__ import annotations

from typing import Any, Dict, List

from services.metrics_service import explain_metrics_by_mode
from services.mock_analysis import DISCLAIMER
from services.qwen_vision import analyze_image_with_qwen
from services.rag_service import answer_question, retrieve_relevant_docs
from services.report_service import generate_markdown_report


def run_image_analysis_agent(uploaded_file: Any) -> Dict[str, Any]:
    """Run image understanding, route selection, and metric explanation."""
    qwen_response = analyze_image_with_qwen(uploaded_file)
    if not qwen_response.get("success"):
        return {
            "success": False,
            "analysis_result": None,
            "error": qwen_response.get("error", "千问图像理解调用失败。"),
        }

    qwen_data = qwen_response.get("data", {})
    image_type = _as_text(qwen_data.get("image_type")) or _infer_image_type(qwen_data)
    image_type_confidence = _as_float(qwen_data.get("image_type_confidence"), default=0.0)
    analysis_focus = _as_list(qwen_data.get("analysis_focus")) or _infer_analysis_focus(image_type, qwen_data)
    metric_mode = _normalize_metric_mode(
        _as_text(qwen_data.get("metric_explanation_mode")) or _infer_metric_mode(image_type, qwen_data)
    )
    suggested_metrics = _as_list(qwen_data.get("suggested_metrics"))
    metric_explanations = explain_metrics_by_mode(metric_mode, suggested_metrics)
    analysis_path = _select_analysis_path(image_type, metric_mode)

    agent_decision = {
        "image_type": image_type,
        "image_type_confidence": image_type_confidence,
        "analysis_path": analysis_path,
        "reason": _build_decision_reason(image_type, analysis_focus, metric_mode),
        "next_steps": _build_next_steps(image_type, metric_mode),
    }

    analysis_result = {
        "image_type": image_type,
        "image_type_confidence": image_type_confidence,
        "image_summary": _as_text(qwen_data.get("image_summary")) or "千问未返回图像整体描述。",
        "analysis_focus": analysis_focus,
        "visible_structures": _as_list(qwen_data.get("visible_structures")),
        "possible_abnormal_regions": _as_list(qwen_data.get("possible_abnormal_regions")),
        "quality_notes": _as_list(qwen_data.get("quality_notes")),
        "suggested_metrics": suggested_metrics,
        "metric_explanations": metric_explanations,
        "limitations": _as_list(qwen_data.get("limitations")),
        "safety_note": _as_text(qwen_data.get("safety_note")) or DISCLAIMER,
        "metric_explanation_mode": metric_mode,
        "agent_decision": agent_decision,
        "qwen_structured_result": qwen_data,
    }

    return {
        "success": True,
        "analysis_result": analysis_result,
        "data": analysis_result,
        "raw_output": qwen_response.get("raw_output", ""),
        "error": "",
    }


def run_image_agent(uploaded_file: Any) -> Dict[str, Any]:
    """Backward-compatible alias for earlier UI code."""
    return run_image_analysis_agent(uploaded_file)


def run_qa_agent(
    question: str,
    analysis_result: Dict[str, object] | None,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, object]:
    """Run local-knowledge-base QA against the current image-analysis context."""
    try:
        answer = answer_question(question, analysis_result)
        sources = [document["file_name"] for document in retrieve_relevant_docs(question)]
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "chat_history": (chat_history or []) + [{"question": question, "answer": answer}],
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "question": question,
            "answer": "",
            "sources": [],
            "chat_history": chat_history or [],
            "error": str(exc),
        }


def run_report_agent(
    analysis_result: Dict[str, object],
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, object]:
    """Generate a markdown report through the Agent orchestration layer."""
    try:
        markdown = generate_markdown_report(analysis_result, chat_history=chat_history)
        return {
            "success": True,
            "report_markdown": markdown,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "report_markdown": "",
            "error": str(exc),
        }


def agent_result_to_app_result(
    agent_data: Dict[str, Any],
    file_name: str,
    image_size: tuple[int, int],
) -> Dict[str, Any]:
    """Convert agent output to the existing Streamlit display shape."""
    width, height = image_size
    quality_notes = _as_list(agent_data.get("quality_notes"))
    metric_explanations = _as_list_of_dicts(agent_data.get("metric_explanations"))
    if not metric_explanations:
        metric_explanations = explain_metrics_by_mode(
            _normalize_metric_mode(_as_text(agent_data.get("metric_explanation_mode"))),
            _as_list(agent_data.get("suggested_metrics")),
        )

    return {
        "analysis_mode": "千问 Agent 图像分析",
        "file_name": file_name,
        "image_size": f"{width} x {height}px",
        "image_type": agent_data.get("image_type", "未明确识别"),
        "image_type_confidence": agent_data.get("image_type_confidence", 0.0),
        "image_quality": "\n".join(quality_notes) if quality_notes else "Agent 未返回图像质量说明。",
        "overview": agent_data.get("image_summary", "Agent 未返回图像概述。"),
        "analysis_focus": _as_list(agent_data.get("analysis_focus")),
        "visible_structures": _as_list(agent_data.get("visible_structures")),
        "abnormal_regions": [
            {
                "region": f"疑似区域 {index}",
                "description": description,
                "confidence": "需人工复核",
            }
            for index, description in enumerate(
                _as_list(agent_data.get("possible_abnormal_regions")),
                start=1,
            )
        ],
        "segmentation_summary": _build_display_summary(agent_data),
        "metrics": [
            {
                "name": item.get("metric", "建议指标"),
                "value": None,
                "explanation": item.get("explanation", "该指标需要结合具体图像和实验任务解释。"),
                "interpretation": item.get("explanation", "该指标需要结合具体图像和实验任务解释。"),
            }
            for item in metric_explanations
        ],
        "quality_notes": quality_notes,
        "suggested_metrics": _as_list(agent_data.get("suggested_metrics")),
        "metric_explanations": metric_explanations,
        "limitations": _as_list(agent_data.get("limitations")),
        "agent_decision": agent_data.get("agent_decision", {}),
        "qwen_structured_result": agent_data.get("qwen_structured_result", {}),
        "disclaimer": agent_data.get("safety_note") or DISCLAIMER,
    }


def _infer_image_type(qwen_data: Dict[str, Any]) -> str:
    combined = " ".join(
        [
            _as_text(qwen_data.get("image_summary")),
            " ".join(_as_list(qwen_data.get("visible_structures"))),
            " ".join(_as_list(qwen_data.get("suggested_metrics"))),
        ]
    ).lower()
    if any(keyword in combined for keyword in ["loss", "accuracy", "曲线", "epoch", "训练"]):
        return "loss 曲线图"
    if any(keyword in combined for keyword in ["mask", "segmentation", "分割", "轮廓"]):
        return "分割结果图"
    if any(keyword in combined for keyword in ["fluorescence", "荧光"]):
        return "荧光显微图像"
    if any(keyword in combined for keyword in ["cell", "细胞", "显微"]):
        return "细胞显微图像"
    return "科研实验图像"


def _infer_analysis_focus(image_type: str, qwen_data: Dict[str, Any]) -> List[str]:
    if "loss" in image_type.lower() or "曲线" in image_type:
        return ["训练趋势", "收敛情况", "过拟合风险"]
    if "分割" in image_type:
        return ["分割边界", "mask 重叠", "漏分割与过分割风险"]
    if "荧光" in image_type:
        return ["荧光强度", "背景噪声", "信噪比"]
    if "细胞" in image_type:
        return ["可见细胞结构", "细胞表型", "图像质量"]
    metrics = _as_list(qwen_data.get("suggested_metrics"))
    return metrics or ["图像内容识别", "质量评估", "后续分析建议"]


def _infer_metric_mode(image_type: str, qwen_data: Dict[str, Any]) -> str:
    combined = " ".join([image_type, " ".join(_as_list(qwen_data.get("suggested_metrics")))]).lower()
    if any(keyword in combined for keyword in ["loss", "accuracy", "曲线", "epoch"]):
        return "curve"
    if any(keyword in combined for keyword in ["dice", "iou", "mask", "分割"]):
        return "segmentation"
    if any(keyword in combined for keyword in ["fluorescence", "荧光", "信噪比"]):
        return "fluorescence"
    if any(keyword in combined for keyword in ["cell", "细胞", "表型"]):
        return "phenotype"
    return "general_quality"


def _normalize_metric_mode(mode: str) -> str:
    normalized = (mode or "").lower()
    if any(keyword in normalized for keyword in ["curve", "loss", "training", "训练", "曲线"]):
        return "curve"
    if any(keyword in normalized for keyword in ["segment", "mask", "dice", "iou", "分割"]):
        return "segmentation"
    if any(keyword in normalized for keyword in ["fluorescence", "荧光", "信噪比"]):
        return "fluorescence"
    if any(keyword in normalized for keyword in ["phenotype", "cell", "表型", "细胞"]):
        return "phenotype"
    return "general_quality"


def _select_analysis_path(image_type: str, metric_mode: str) -> str:
    if metric_mode == "curve":
        return "training_curve_analysis"
    if metric_mode == "segmentation":
        return "segmentation_evaluation"
    if metric_mode == "fluorescence":
        return "fluorescence_quantification"
    if metric_mode == "phenotype":
        return "cell_quantification"
    if "细胞" in image_type:
        return "cell_quantification"
    return "general_quality_review"


def _build_decision_reason(image_type: str, analysis_focus: List[str], metric_mode: str) -> str:
    focus_text = "、".join(analysis_focus) if analysis_focus else "图像结构与质量"
    return (
        f"Agent 将图像识别为“{image_type}”，主要关注“{focus_text}”。"
        f"因此选择 `{metric_mode}` 指标解释模式，避免在不适合的图像类型上套用分割指标。"
    )


def _build_next_steps(image_type: str, metric_mode: str) -> str:
    if metric_mode == "curve":
        return ["结合训练日志、验证集曲线和学习率设置判断收敛与过拟合风险。"]
    if metric_mode == "segmentation":
        return ["补充 ground truth mask 和 prediction mask。", "再计算 Dice、IoU、Precision、Recall。"]
    if metric_mode == "fluorescence":
        return ["统一曝光、增益、背景扣除和 ROI 规则。", "再做荧光强度定量比较。"]
    if metric_mode == "phenotype":
        return ["结合细胞面积、数量、密度、聚集程度和重复样本做表型比较。"]
    return ["先进行人工复核、图像质量检查和任务目标确认。", "再选择合适的定量指标。"]


def _build_display_summary(agent_data: Dict[str, Any]) -> str:
    decision = agent_data.get("agent_decision", {})
    analysis_path = decision.get("analysis_path", "通用科研图像质量与结构分析路径")
    return (
        f"Agent 已选择“{analysis_path}”。当前阶段只进行图像理解、路径选择和指标解释，"
        "不执行真实分割、不计算真实分割指标。"
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(str(item) for item in value)
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_list_of_dicts(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, dict):
            items.append({str(key): str(val) for key, val in item.items()})
    return items
