from __future__ import annotations

import streamlit as st
from services.ui_components import (
    init_state,
    render_analysis_section,
    render_header,
    render_qa_section,
    render_report_section,
    render_upload_section,
)

st.set_page_config(
    page_title="科研图像智能分析 Agent",
    page_icon="🔬",
    layout="wide",
)


def main() -> None:
    init_state()
    render_header()

    upload_col, analysis_col = st.columns([0.95, 1.25], gap="large")
    with upload_col:
        uploaded_file, image = render_upload_section()
    with analysis_col:
        render_analysis_section(uploaded_file, image)

    st.divider()
    qa_col, report_col = st.columns([1, 1], gap="large")
    with qa_col:
        render_qa_section()
    with report_col:
        render_report_section()


if __name__ == "__main__":
    main()
