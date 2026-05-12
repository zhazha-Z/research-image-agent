"""Streamlit UI sections for the research image analysis MVP."""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from services.agent_orchestrator import (
    agent_result_to_app_result,
    run_image_analysis_agent,
    run_qa_agent,
    run_report_agent,
)
from services.mock_analysis import DISCLAIMER


def init_state() -> None:
    """Initialize Streamlit session state used across the MVP flow."""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "report_markdown" not in st.session_state:
        st.session_state.report_markdown = ""


def render_header() -> None:
    st.title("基于多模态大模型的科研图像智能分析 Agent")
    st.caption("面向生物、林学与细胞生物学科研图像的千问 AI 图像分析原型")
    st.info(DISCLAIMER)


def render_upload_section() -> tuple[object | None, Image.Image | None]:
    st.subheader("1. 图像上传与预览")
    uploaded_file = st.file_uploader(
        "上传 JPG 或 PNG 科研图像",
        type=["jpg", "jpeg", "png"],
        help="上传后将使用千问 AI 图像理解能力进行分析。",
    )

    if uploaded_file is None:
        st.warning("请先上传一张细胞显微图像或实验图像。")
        return None, None

    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(image, caption=f"图像预览：{uploaded_file.name}", use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("宽度", f"{image.width}px")
    col_b.metric("高度", f"{image.height}px")
    col_c.metric("格式", uploaded_file.type or "未知")

    return uploaded_file, image


def render_analysis_section(uploaded_file: object | None, image: Image.Image | None) -> None:
    st.subheader("2. 千问 AI 图像分析")
    disabled = uploaded_file is None or image is None
    st.caption("系统会调用阿里云百炼 DashScope 千问图像理解模型进行科研辅助分析。")

    if st.button("开始 AI 分析", type="primary", disabled=disabled, use_container_width=True):
        st.session_state.analysis_error = ""
        st.session_state.report_markdown = ""
        with st.spinner("Agent 正在识别图像类型、选择分析路径并生成结构化反馈..."):
            agent_response = run_image_analysis_agent(uploaded_file)
        if agent_response.get("success"):
            st.session_state.analysis_result = agent_result_to_app_result(
                agent_response["analysis_result"],
                file_name=uploaded_file.name,
                image_size=image.size,
            )
            st.success("Agent 图像分析完成。")
        else:
            st.session_state.analysis_error = agent_response.get("error", "Agent 图像分析失败。")
            st.session_state.analysis_result = None

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)
        st.info("请检查 DASHSCOPE_API_KEY、网络连接或 QWEN_VISION_MODEL 模型名称后重试。")

    result = st.session_state.analysis_result
    if not result:
        st.caption("上传图像后点击“开始 AI 分析”，这里会展示结构分析、异常区域、图像质量和指标建议。")
        return

    if result.get("agent_decision"):
        _render_agent_decision(result)

    st.markdown("#### 图像概述")
    st.write(result["overview"])

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### 可见结构")
        for item in result["visible_structures"]:
            st.markdown(f"- {item}")

        st.markdown("#### 疑似异常区域")
        for item in result["abnormal_regions"]:
            st.markdown(
                f"- **{item['region']}**（置信度：{item['confidence']}）：{item['description']}"
            )

        st.markdown("#### 分割结果说明")
        st.write(result["segmentation_summary"])

        if result.get("quality_notes"):
            st.markdown("#### 图像质量说明")
            for item in result["quality_notes"]:
                st.markdown(f"- {item}")

    with right:
        st.markdown("#### 建议分析指标")
        metric_cols = st.columns(2)
        for index, metric in enumerate(result["metrics"]):
            with metric_cols[index % 2]:
                value = metric.get("value")
                st.metric(metric["name"], f"{value:.2f}" if isinstance(value, float) else "建议关注")
                st.caption(metric["interpretation"])

        with st.expander("每个指标的通俗解释", expanded=True):
            for metric in result["metrics"]:
                st.markdown(f"**{metric['name']}**：{metric['explanation']}")

        if result.get("limitations"):
            with st.expander("不确定性和局限性", expanded=True):
                for item in result["limitations"]:
                    st.markdown(f"- {item}")

        if result.get("qwen_structured_result"):
            with st.expander("千问返回的结构化 JSON", expanded=False):
                st.json(result["qwen_structured_result"])

    st.markdown("#### 安全提示")
    st.warning(result.get("disclaimer") or DISCLAIMER)


def render_qa_section() -> None:
    st.subheader("3. 追问与本地知识库")
    st.caption("示例：这个异常区域是什么意思？Dice 和 IoU 有什么区别？这个分割效果好不好？有哪些相关论文方法？")

    question = st.text_input("输入你的追问", placeholder="例如：Dice 和 IoU 有什么区别？")
    ask_clicked = st.button("提交追问", disabled=not question.strip(), use_container_width=True)

    if ask_clicked:
        qa_response = run_qa_agent(
            question,
            st.session_state.analysis_result,
            chat_history=st.session_state.chat_history,
        )
        if qa_response.get("success"):
            st.session_state.chat_history = qa_response["chat_history"]
        else:
            st.session_state.chat_history.append(
                {"question": question, "answer": f"问答失败：{qa_response.get('error', '未知错误')}"}
            )

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
    else:
        st.caption("追问回答当前由 `knowledge_base/` 本地 markdown 知识库检索生成。")


def render_report_section() -> None:
    st.subheader("4. 生成 Markdown 实验报告")
    disabled = st.session_state.analysis_result is None

    if st.button("生成实验报告", disabled=disabled, use_container_width=True):
        report_response = run_report_agent(
            st.session_state.analysis_result,
            chat_history=st.session_state.chat_history,
        )
        if report_response.get("success"):
            st.session_state.report_markdown = report_response["report_markdown"]
        else:
            st.error(f"报告生成失败：{report_response.get('error', '未知错误')}")

    if not st.session_state.report_markdown:
        st.caption("完成分析后可生成包含实验目的、图像结果、指标解读、文献方法与局限性的 Markdown 报告。")
        return

    st.markdown(st.session_state.report_markdown)
    st.download_button(
        label="下载 Markdown 报告",
        data=st.session_state.report_markdown,
        file_name="research_image_analysis_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _render_agent_decision(result: dict) -> None:
    decision = result.get("agent_decision") or {}
    st.markdown("#### Agent 判断结果")
    col_a, col_b = st.columns(2)
    col_a.markdown(f"**图像类型**：{decision.get('image_type', result.get('image_type', '未明确识别'))}")
    confidence = decision.get("image_type_confidence", result.get("image_type_confidence", 0.0))
    col_b.markdown(f"**图像类型置信度**：{float(confidence):.2f}")
    st.markdown(f"**分析路径**：{decision.get('analysis_path', '未明确选择')}")

    focus_items = result.get("analysis_focus") or []
    if focus_items:
        st.markdown("**分析重点**")
        for item in focus_items:
            st.markdown(f"- {item}")

    if decision.get("reason"):
        st.markdown(f"**路径选择原因**：{decision['reason']}")
    if decision.get("next_steps"):
        st.markdown("**建议下一步**")
        next_steps = decision["next_steps"]
        if isinstance(next_steps, list):
            for item in next_steps:
                st.markdown(f"- {item}")
        else:
            st.markdown(f"- {next_steps}")
