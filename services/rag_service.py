"""Mock RAG service backed by a small in-memory literature knowledge base."""

from __future__ import annotations

from typing import Dict, List

from services.mock_analysis import DISCLAIMER


MOCK_LITERATURE_KB: List[Dict[str, str]] = [
    {
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "method": "U-Net",
        "summary": "经典编码器-解码器结构，适用于小样本生物医学图像分割，是细胞、组织和显微图像分割中的常用基线。",
    },
    {
        "title": "Attention U-Net: Learning Where to Look for the Pancreas",
        "method": "Attention U-Net",
        "summary": "在跳跃连接中加入注意力机制，帮助模型聚焦关键结构，可迁移到细胞边界和弱信号区域分析。",
    },
    {
        "title": "Segment Anything",
        "method": "SAM",
        "summary": "通用提示式分割框架，可通过点、框或掩膜提示生成候选分割结果，适合原型阶段做人机协同标注。",
    },
    {
        "title": "Cellpose: a generalist algorithm for cellular segmentation",
        "method": "Cellpose",
        "summary": "面向细胞形态的通用分割方法，对多种细胞类型和显微成像条件具有较好泛化能力。",
    },
]


def answer_question(question: str, analysis_result: Dict[str, object] | None = None) -> str:
    """Answer a user follow-up with deterministic mock knowledge."""
    normalized = question.strip().lower()
    if not normalized:
        return f"请输入一个关于图像、指标或文献方法的问题。{DISCLAIMER}"

    if any(keyword in normalized for keyword in ["dice", "iou", "区别"]):
        answer = (
            "Dice 和 IoU 都衡量分割结果与参考标注的重叠程度。Dice 更强调两者的整体重合，数值通常会比 IoU 高；"
            "IoU 使用交集除以并集，对多分或漏分更敏感，因此判断更严格。若 Dice=0.86、IoU=0.76，通常可认为主体区域分割较可靠，"
            "但边界仍需要人工复核。"
        )
    elif any(keyword in normalized for keyword in ["异常", "区域", "高亮", "是什么意思"]):
        answer = (
            "mock 结果中的疑似异常区域主要表示局部亮度、纹理或形态与周围区域不同。它可能对应真实生物学现象，"
            "例如细胞聚集、荧光信号增强、组织结构变化；也可能只是样本制备、染色不均、曝光或噪声带来的成像伪影。"
            "建议结合实验分组、原始通道图、重复样本和人工标注进一步判断。"
        )
    elif any(keyword in normalized for keyword in ["好不好", "效果", "分割"]):
        answer = (
            "从 mock 指标看，Dice=0.86、IoU=0.76、Precision=0.89、Recall=0.82，说明主体目标识别较好，误检相对少，"
            "但仍可能存在少量漏检和边界偏差。对于作品集 MVP，可以把它解释为“可用于初筛和辅助复核，但不能替代人工质控”。"
        )
    elif any(keyword in normalized for keyword in ["论文", "文献", "方法", "相关"]):
        answer = "可参考以下 mock 文献方法：\n\n" + "\n".join(
            f"- **{item['method']}**：{item['summary']}（{item['title']}）"
            for item in MOCK_LITERATURE_KB
        )
    else:
        answer = (
            "这是 mock 知识库回答：当前图像可以从目标结构、疑似异常区域、分割边界和指标可靠性四个角度解读。"
            "如果问题涉及实验结论，建议结合样本来源、染色方案、成像参数、人工标注和重复实验共同判断。"
        )

    if analysis_result:
        file_name = analysis_result.get("file_name", "当前图像")
        answer = f"针对 `{file_name}`：{answer}"

    return f"{answer}\n\n> {DISCLAIMER}"


def get_literature_references() -> List[Dict[str, str]]:
    """Return mock literature references for reports."""
    return MOCK_LITERATURE_KB
