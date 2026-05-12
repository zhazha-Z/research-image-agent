# 基于多模态大模型的科研图像智能分析 Agent

这是一个使用 Python + Streamlit 开发的科研图像智能分析 Agent 原型，面向生物、林学和细胞生物学科研人员，演示科研图像上传、预览、智能分析、指标解释、追问问答和 Markdown 实验报告生成的完整产品流程。

当前页面默认使用阿里云百炼 DashScope 千问图像理解，基于 OpenAI 兼容接口调用 `qwen-vl-plus` 等视觉模型；API Key 从环境变量 `DASHSCOPE_API_KEY` 读取，不会在代码中硬编码。项目暂时不接入真实分割模型，不接入真实向量数据库。

所有输出仅用于科研辅助分析，不构成医学诊断。真实 AI 分析结果仍需结合实验背景、人工标注和重复实验复核。

## 项目结构

```text
.
├── app.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── services
    ├── mock_analysis.py
    ├── agent_orchestrator.py
    ├── metrics_service.py
    ├── export_service.py
    ├── qwen_vision.py
    ├── rag_service.py
    ├── report_service.py
    ├── ui_components.py
    └── vision_analysis.py
```

职责划分：

- `app.py`：Streamlit 应用入口，只负责页面配置和主流程布局。
- `services/ui_components.py`：页面区块渲染，包括上传预览、分析展示、追问和报告区。
- `services/agent_orchestrator.py`：编排图像类型识别、分析路径选择、指标解释模式选择和结构化结果反馈。
- `services/metrics_service.py`：按分割、曲线、荧光、表型和通用质量模式提供指标解释。
- `services/export_service.py`：导出 Markdown 报告配套结果、分割 mask、面积统计 CSV 和完整分析 JSON。
- `services/mock_analysis.py`：本地演示结果生成器，仅保留给开发调试使用，正式页面不直接调用。
- `services/qwen_vision.py`：通过 DashScope OpenAI 兼容接口调用千问图像理解模型。
- `services/rag_service.py`：读取 `knowledge_base/` 下的 markdown 文件，提供本地知识库关键词检索和追问回答。
- `services/report_service.py`：根据分析结果生成 Markdown 实验报告。
- `services/vision_analysis.py`：上一版 OpenAI 视觉模型服务封装，当前页面主流程使用千问图像理解。

## 安装依赖

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖说明：

- `streamlit`：构建交互式 Web 原型。
- `pillow`：读取和预览 JPG/PNG 图像。
- `openai`：使用 OpenAI Python SDK 调用 DashScope OpenAI 兼容接口。
- `python-dotenv`：本地开发时从 `.env` 读取环境变量。

## 配置 API Key

千问 AI 分析模式需要配置环境变量 `DASHSCOPE_API_KEY`。不要把 API Key 写进代码，也不要提交到 GitHub。

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
QWEN_VISION_MODEL=qwen-vl-plus
```

也可以直接在终端环境中设置：

```bash
export DASHSCOPE_API_KEY=your_dashscope_api_key_here
export QWEN_VISION_MODEL=qwen-vl-plus
```

## 启动方式

```bash
streamlit run app.py
```

启动后在浏览器中打开 Streamlit 提供的本地地址，上传 JPG 或 PNG 图像，然后点击“开始 AI 分析”即可体验完整流程。当前页面默认调用千问 AI 图像分析；没有配置 `DASHSCOPE_API_KEY` 时，真实分析无法使用，页面会提示错误原因且不会崩溃。

## 当前功能

- 上传 JPG/PNG 科研图像。
- 显示图像预览、宽度、高度和文件格式。
- 上传图片后点击“开始 AI 分析”即可调用千问图像理解。
- Agent 会自动识别图像类型，选择分析路径，并给出路径选择原因和下一步建议。
- 对细胞显微图像或荧光显微图像，可运行基础 Otsu 阈值分割与连通域面积统计。
- 千问 AI 分析使用细胞显微图像和科研实验图像的图像理解 Prompt，返回结构化 JSON。
- 输出图像概述、可见结构、疑似异常区域、图像质量说明、建议分析指标和局限性。
- 如果没有配置 `DASHSCOPE_API_KEY`，或网络、模型名称、API 调用异常，真实分析无法使用。
- 支持用户输入追问，并基于 `knowledge_base/` 本地 markdown 知识库进行关键词检索回答。
- 点击“生成实验报告”后输出 Markdown 报告。
- 支持下载 Markdown 报告。
- 如果运行了基础分割统计，Markdown 报告会追加基础分割与面积统计结果。
- V1.3 支持下载完整分析结果 JSON、分割 mask PNG 和面积统计 CSV。
- 所有关键输出均提示：仅用于科研辅助分析，不构成医学诊断。

## 结果导出

V1.3 增强了报告导出与分析结果归档：

- 支持下载 Markdown 实验报告。
- 支持下载基础分割 mask 图。
- 支持下载面积统计 CSV，包含每个目标面积和 summary 统计。
- 支持下载完整分析结果 JSON，包含图像分析结果、追问记录和可选分割统计。

运行中生成的归档文件会保存到 `outputs/` 目录。该目录用于本地运行结果缓存，具体实验输出文件不会提交到 GitHub；仓库只保留 `outputs/.gitkeep`。

## 追问示例

- 这个异常区域是什么意思？
- Dice 和 IoU 有什么区别？
- 这个分割效果好不好？
- 有哪些相关论文方法？

## 本地知识库

当前 RAG 是轻量级“本地 markdown 知识库关键词检索”，还不是向量数据库 RAG。知识库目录为 `knowledge_base/`，系统会读取该目录下的 `.md` 文件，并根据用户问题进行简单关键词匹配，返回相关内容和来源文件名。

你可以直接在 `knowledge_base/` 中新增或编辑 `.md` 文件来扩展知识库。建议每个文件包含标题、适用场景、核心概念、常见问题、局限性和可回答问题示例。后续可以升级为 Chroma、FAISS 或其他向量数据库检索，以支持语义检索、文献切片和更稳定的上下文召回。

## 报告结构

生成的 Markdown 报告包含：

- 实验目的
- 图像数据说明
- 图像分析结果
- 指标解读
- 异常区域解释
- 相关文献方法参考
- 初步结论
- 局限性与后续建议

## 后续升级路线

1. 接入真实多模态大模型，替换 `services/mock_analysis.py` 中的 mock 图像语义分析。
2. 接入真实细胞或组织图像分割模型，生成 mask、边界和异常区域可视化。
3. 引入人工标注对照，计算真实 Dice、IoU、Precision、Recall。
4. 将当前本地 markdown 关键词检索升级为 Chroma、FAISS 或其他向量数据库 RAG。
5. 增加 PDF/DOCX 报告导出、报告模板配置和实验元数据填写。
6. 增加多图批量分析、结果对比和项目级实验记录管理。
