# 基于多模态大模型的科研图像智能分析 Agent

这是一个使用 Python + Streamlit 开发的科研图像智能分析 Agent 原型，面向生物、林学和细胞生物学科研人员，演示科研图像上传、预览、智能分析、指标解释、追问问答和 Markdown 实验报告生成的完整产品流程。

项目默认保留 mock 分析模式，不接入真实分割模型，不接入真实向量数据库。当前已新增阿里云百炼 DashScope 千问图像理解模式，使用 OpenAI 兼容接口调用 `qwen-vl-plus` 等视觉模型；API Key 从环境变量 `DASHSCOPE_API_KEY` 读取，不会在代码中硬编码。

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
    ├── qwen_vision.py
    ├── rag_service.py
    ├── report_service.py
    ├── ui_components.py
    └── vision_analysis.py
```

职责划分：

- `app.py`：Streamlit 应用入口，只负责页面配置和主流程布局。
- `services/ui_components.py`：页面区块渲染，包括上传预览、分析展示、追问和报告区。
- `services/mock_analysis.py`：生成 mock 图像概述、可见结构、异常区域、分割说明和指标解释。
- `services/qwen_vision.py`：通过 DashScope OpenAI 兼容接口调用千问图像理解模型。
- `services/rag_service.py`：提供 mock 文献知识库和追问回答。
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

启动后在浏览器中打开 Streamlit 提供的本地地址，上传 JPG 或 PNG 图像即可体验完整流程。没有配置 `DASHSCOPE_API_KEY` 时，仍可选择“mock 分析”；如果选择“千问 AI 分析”，页面会提示错误原因并自动回退到 mock 分析。

## 当前功能

- 上传 JPG/PNG 科研图像。
- 显示图像预览、宽度、高度和文件格式。
- 可在页面中切换“mock 分析”和“千问 AI 分析”。
- 点击“开始分析”后展示图像分析结果。
- 千问 AI 分析使用细胞显微图像和科研实验图像的图像理解 Prompt，返回结构化 JSON。
- 输出图像概述、可见结构、疑似异常区域、图像质量说明、建议分析指标和局限性。
- mock 模式展示 Dice、IoU、Precision、Recall 指标及通俗解释。
- 支持用户输入追问，并基于内置 mock 文献知识库回答。
- 点击“生成实验报告”后输出 Markdown 报告。
- 支持下载 Markdown 报告。
- 所有关键输出均提示：仅用于科研辅助分析，不构成医学诊断。

## 追问示例

- 这个异常区域是什么意思？
- Dice 和 IoU 有什么区别？
- 这个分割效果好不好？
- 有哪些相关论文方法？

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
4. 接入文献向量数据库或检索服务，替换 `services/rag_service.py` 中的内存 mock 知识库。
5. 增加 PDF/DOCX 报告导出、报告模板配置和实验元数据填写。
6. 增加多图批量分析、结果对比和项目级实验记录管理。
