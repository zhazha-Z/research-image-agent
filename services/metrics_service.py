"""Metric explanation helpers selected by the image-analysis agent."""

from __future__ import annotations

from typing import Dict, Iterable, List


def explain_segmentation_metrics(metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Explain segmentation metrics only when a segmentation-evaluation path is selected."""
    explanations = {
        "dice": "Dice 衡量 prediction mask 与 ground truth mask 的重叠程度，越接近 1 表示整体重叠越充分。",
        "iou": "IoU 使用交集除以并集，比 Dice 更严格，适合观察过分割和漏分割带来的影响。",
        "precision": "Precision 关注预测为目标的区域中有多少确实属于目标，可帮助判断误检是否偏多。",
        "recall": "Recall 关注真实目标中有多少被预测出来，可帮助判断漏检是否偏多。",
    }
    return _select_explanations(metrics, explanations, fallback_keys=["dice", "iou", "precision", "recall"])


def explain_curve_metrics(metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Explain common training or loss-curve metrics."""
    explanations = {
        "loss": "Loss 曲线用于观察模型训练目标是否下降，持续下降通常说明优化在进行，但不等同于泛化能力一定提升。",
        "train loss": "Train loss 反映训练集上的拟合情况，过低但验证集不改善时可能存在过拟合。",
        "val loss": "Validation loss 更适合观察泛化趋势，若持续上升或震荡，需要检查学习率、数据划分和正则化。",
        "accuracy": "Accuracy 适合类别相对均衡的分类任务，类别不平衡时应结合 Precision、Recall 或 F1。",
    }
    return _select_explanations(metrics, explanations, fallback_keys=["loss", "train loss", "val loss"])


def explain_fluorescence_metrics(metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Explain fluorescence-analysis metrics."""
    explanations = {
        "荧光强度": "荧光强度可用于比较信号水平，但需要统一曝光、增益、背景扣除和归一化流程。",
        "背景噪声": "背景噪声会影响阈值判断和阳性区域统计，应在定量前进行背景估计或扣除。",
        "信噪比": "信噪比用于描述目标信号相对于背景噪声的可分辨程度，低信噪比会降低解释可靠性。",
        "阳性面积": "阳性面积可描述超过阈值的信号范围，但阈值策略应在实验组之间保持一致。",
    }
    return _select_explanations(metrics, explanations, fallback_keys=["荧光强度", "背景噪声", "信噪比"])


def explain_phenotype_metrics(metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Explain cell phenotype metrics."""
    explanations = {
        "细胞面积": "细胞面积可反映形态变化，但需要稳定分割边界和一致的 ROI 规则。",
        "细胞数量": "细胞数量适合比较密度或增殖趋势，应结合采样视野和重复实验。",
        "细胞密度": "细胞密度需要明确单位面积和采样区域，避免视野选择偏差。",
        "聚集程度": "聚集程度可提示空间分布变化，但不能单独推断机制，需要结合实验条件。",
        "形态": "形态指标包括圆度、长宽比、周长等，适合描述处理组之间的表型差异。",
    }
    return _select_explanations(metrics, explanations, fallback_keys=["细胞面积", "细胞数量", "细胞密度", "聚集程度"])


def explain_general_quality_metrics(metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Explain general image-quality or exploratory metrics."""
    explanations = {
        "清晰度": "清晰度影响边界判断和结构识别，模糊图像会降低分割和解释稳定性。",
        "对比度": "对比度影响目标与背景的可分辨程度，对阈值分割和人工复核都很重要。",
        "噪声": "噪声会干扰目标识别和强度统计，必要时应进行质控或预处理。",
        "曝光": "曝光不足或过曝都会影响信号解释，过曝还可能造成像素饱和。",
    }
    return _select_explanations(metrics, explanations, fallback_keys=["清晰度", "对比度", "噪声", "曝光"])


def explain_metrics_by_mode(mode: str, metrics: Iterable[str]) -> List[Dict[str, str]]:
    """Dispatch metric explanations by agent-selected explanation mode."""
    normalized_mode = (mode or "").lower()
    if any(keyword in normalized_mode for keyword in ["segment", "mask", "分割"]):
        return explain_segmentation_metrics(metrics)
    if any(keyword in normalized_mode for keyword in ["curve", "loss", "training", "训练", "曲线"]):
        return explain_curve_metrics(metrics)
    if any(keyword in normalized_mode for keyword in ["fluorescence", "荧光"]):
        return explain_fluorescence_metrics(metrics)
    if any(keyword in normalized_mode for keyword in ["phenotype", "cell", "表型", "细胞"]):
        return explain_phenotype_metrics(metrics)
    return explain_general_quality_metrics(metrics)


def _select_explanations(
    metrics: Iterable[str],
    explanation_map: Dict[str, str],
    fallback_keys: List[str],
) -> List[Dict[str, str]]:
    selected = []
    metric_names = [str(metric) for metric in metrics if str(metric).strip()]
    keys_to_scan = metric_names or fallback_keys

    for metric_name in keys_to_scan:
        normalized_metric = metric_name.lower()
        for key, explanation in explanation_map.items():
            if key.lower() in normalized_metric or normalized_metric in key.lower():
                selected.append({"metric": metric_name, "explanation": explanation})
                break

    if selected:
        return _dedupe_explanations(selected)

    return [
        {"metric": key, "explanation": explanation_map[key]}
        for key in fallback_keys
        if key in explanation_map
    ]


def _dedupe_explanations(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []
    for item in items:
        key = item["metric"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
