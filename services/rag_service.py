"""Local markdown knowledge-base retrieval for follow-up questions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from services.mock_analysis import DISCLAIMER


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_DIR = PROJECT_ROOT / "knowledge_base"

DOMAIN_TERMS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "ground truth",
    "mask",
    "prediction",
    "阈值",
    "分割",
    "细胞",
    "显微",
    "荧光",
    "强度",
    "曝光",
    "信噪比",
    "背景",
    "表型",
    "形态",
    "面积",
    "密度",
    "聚集",
    "模糊",
    "过曝",
    "噪声",
    "对比度",
    "质量",
    "u-net",
    "unet",
    "cellpose",
    "sam",
]

STOP_TOKENS = {
    "什么",
    "怎么",
    "哪些",
    "是否",
    "如何",
    "可以",
    "能不",
    "不能",
    "影响",
    "问题",
    "相关",
    "无关",
    "完全",
}


def load_knowledge_base(kb_dir: str = "knowledge_base") -> List[Dict[str, str]]:
    """Read all markdown documents from the local knowledge base directory."""
    kb_path = Path(kb_dir)
    if not kb_path.is_absolute():
        kb_path = PROJECT_ROOT / kb_path

    if not kb_path.exists():
        return []

    documents = []
    for path in sorted(kb_path.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        documents.append(
            {
                "title": _extract_title(content, path.stem),
                "file_name": path.name,
                "content": content,
            }
        )
    return documents


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese and English text with simple local rules."""
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-+.]*", normalized)

    cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for sequence in cjk_sequences:
        tokens.append(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(max(len(sequence) - 1, 0)))
        tokens.extend(sequence[index : index + 3] for index in range(max(len(sequence) - 2, 0)))

    for term in DOMAIN_TERMS:
        if term in normalized:
            tokens.append(term)

    return tokens


def retrieve_relevant_docs(question: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Retrieve the most relevant markdown documents with keyword scoring."""
    query_tokens = set(tokenize(question)) - STOP_TOKENS
    if not _has_domain_signal(question):
        return []
    if not query_tokens:
        return []

    scored_docs = []
    for document in load_knowledge_base():
        title_tokens = set(tokenize(document["title"]))
        file_tokens = set(tokenize(document["file_name"].replace("_", " ")))
        content_tokens = set(tokenize(document["content"]))

        score = 0
        score += 4 * len(query_tokens & title_tokens)
        score += 3 * len(query_tokens & file_tokens)
        score += len(query_tokens & content_tokens)

        if score > 0:
            scored_docs.append({**document, "score": score})

    scored_docs.sort(key=lambda item: item["score"], reverse=True)
    max_score = scored_docs[0]["score"] if scored_docs else 0
    min_score = max(2, max_score * 0.4)
    filtered_docs = [document for document in scored_docs if document["score"] >= min_score]
    return filtered_docs[: max(top_k, 1)]


def answer_question(question: str, analysis_result: Dict[str, object] | None = None) -> str:
    """Answer a follow-up question using local markdown retrieval and image context."""
    clean_question = question.strip()
    if not clean_question:
        return f"请输入一个关于图像、指标或文献方法的问题。\n\n> {DISCLAIMER}"

    docs = retrieve_relevant_docs(clean_question)
    if not docs:
        return _format_no_evidence_answer(clean_question, analysis_result)

    direct_answer = _build_direct_answer(clean_question, docs)
    image_basis = _build_image_basis(analysis_result)
    kb_basis = _build_knowledge_basis(docs)
    uncertainty = (
        "当前回答基于本地 markdown 知识库的关键词检索和当前图像分析摘要，"
        "不等同于系统检索了真实论文数据库，也不能替代人工标注、实验记录和统计验证。"
    )
    next_steps = _build_next_steps(docs)
    source_files = ", ".join(document["file_name"] for document in docs)

    return f"""**直接回答**

{direct_answer}

**图像分析依据**

{image_basis}

**知识库依据**

{kb_basis}

**不确定性说明**

{uncertainty}

**下一步建议**

{next_steps}

**来源文件**

{source_files}

> {DISCLAIMER}"""


def get_literature_references() -> List[Dict[str, str]]:
    """Return local knowledge-base references for the existing report generator."""
    references = []
    for document in load_knowledge_base():
        summary = _first_meaningful_line(document["content"])
        references.append(
            {
                "title": document["title"],
                "method": document["file_name"],
                "summary": summary,
            }
        )
    return references


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _has_domain_signal(text: str) -> bool:
    normalized = text.lower()
    return any(term in normalized for term in DOMAIN_TERMS)


def _build_direct_answer(question: str, docs: List[Dict[str, str]]) -> str:
    snippets = [_extract_relevant_snippet(question, document["content"]) for document in docs]
    joined_snippets = "\n".join(f"- {snippet}" for snippet in snippets if snippet)
    return (
        "根据本地知识库文件，和该问题最相关的要点如下：\n"
        f"{joined_snippets}"
    )


def _build_image_basis(analysis_result: Dict[str, object] | None) -> str:
    if not analysis_result:
        return "当前尚未提供图像分析结果，因此只能依据本地知识库进行通用科研解释。"

    overview = str(analysis_result.get("overview") or "当前图像分析未返回整体概述。")
    structures = analysis_result.get("visible_structures") or []
    abnormalities = analysis_result.get("abnormal_regions") or []
    quality = analysis_result.get("image_quality") or ""

    structure_text = "；".join(str(item) for item in structures[:3]) or "未返回明确可见结构。"
    abnormal_text = "；".join(
        str(item.get("description", item)) if isinstance(item, dict) else str(item)
        for item in abnormalities[:2]
    ) or "未返回明确疑似异常区域。"

    return (
        f"- 图像概述：{overview}\n"
        f"- 可见结构：{structure_text}\n"
        f"- 疑似异常区域：{abnormal_text}\n"
        f"- 图像质量信息：{quality or '未返回明确图像质量说明。'}"
    )


def _build_knowledge_basis(docs: List[Dict[str, str]]) -> str:
    lines = []
    for document in docs:
        snippet = _extract_relevant_snippet(document["title"], document["content"])
        lines.append(f"- `{document['file_name']}`：{snippet}")
    return "\n".join(lines)


def _build_next_steps(docs: List[Dict[str, str]]) -> str:
    file_names = {document["file_name"] for document in docs}
    suggestions = [
        "结合原始图像、实验分组和重复样本进行人工复核。",
        "如果需要定量结论，应补充分割 mask、人工标注或统一采集条件。",
    ]
    if "dice_iou_metrics.md" in file_names:
        suggestions.append("若要计算 Dice、IoU、Precision 或 Recall，请准备 ground truth mask 和 prediction mask。")
    if "fluorescence_analysis.md" in file_names:
        suggestions.append("若要比较荧光强度，请统一曝光、增益、背景扣除和归一化流程。")
    if "microscopy_image_quality.md" in file_names:
        suggestions.append("若图像存在模糊、过曝或噪声，建议先做质量筛查，再进行分割或解释。")
    return "\n".join(f"- {suggestion}" for suggestion in suggestions)


def _format_no_evidence_answer(
    question: str,
    analysis_result: Dict[str, object] | None,
) -> str:
    return f"""**直接回答**

当前本地知识库未找到足够依据，建议补充相关文献或方法说明。

**图像分析依据**

{_build_image_basis(analysis_result)}

**知识库依据**

未检索到相关 markdown 文件。

**不确定性说明**

该问题超出了当前本地知识库覆盖范围，不能据此给出可靠科研判断。

**下一步建议**

- 在 `knowledge_base/` 中补充与“{question}”相关的实验方法、指标解释或文献笔记。
- 补充后重新提问，系统会基于本地 markdown 知识库再次检索。

**来源文件**

无

> {DISCLAIMER}"""


def _extract_relevant_snippet(query: str, content: str) -> str:
    query_tokens = set(tokenize(query))
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip() and not paragraph.strip().startswith("#")
    ]
    if not paragraphs:
        return _first_meaningful_line(content)

    best_paragraph = max(
        paragraphs,
        key=lambda paragraph: len(query_tokens & set(tokenize(paragraph))),
    )
    return _compact_text(best_paragraph)


def _first_meaningful_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return _compact_text(stripped)
    return "本地知识库条目。"


def _compact_text(text: str, max_length: int = 220) -> str:
    without_bullets = re.sub(r"(?m)^\s*-\s*", "", text)
    compacted = re.sub(r"\s+", " ", without_bullets).strip()
    if len(compacted) <= max_length:
        return compacted
    return compacted[: max_length - 1].rstrip() + "…"
