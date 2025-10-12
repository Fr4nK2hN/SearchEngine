[English](#english) | [中文](#中文)

---

# Modern Search Engine for User Behavior Analysis

This project is a modern search engine designed to investigate how different types of non-relevant search results impact user behavior and search efficiency. By manipulating the coherence and relevance of results, this system allows for the analysis of user interaction patterns, such as time spent on search and query abandonment.

## Features

- **Two-Stage Search:** Utilizes a "recall + re-rank" architecture with Elasticsearch for initial candidate retrieval and a `sentence-transformer` cross-encoder model for fine-grained re-ranking.
- **User Behavior Tracking:** Logs critical user interactions, including queries, result clicks, and query abandonment, to a local file (`events.log`).
- **Experimental Framework:** Allows for A/B testing by providing different "modes" of search results, such as an optimal mode and a "degraded relevance" mode.
- **Containerized Environment:** Uses Docker to manage the Elasticsearch service, ensuring a consistent and reproducible setup.

## Tech Stack

- **Backend:** Flask
- **Search:** Elasticsearch 8.x
- **Re-ranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Frontend:** Vanilla JavaScript, HTML, CSS
- **Environment:** Docker, Python 3

## Project Structure

```
.SearchEngine/
├── app.py                  # Main Flask application with search and logging endpoints
├── data_preprocessor.py    # Script to download and process the MS MARCO dataset
├── docker-compose.yml      # Docker configuration for Elasticsearch
├── requirements.txt        # Python dependencies
├── msmarco_docs.json       # Processed documents for indexing
├── events.log              # Log file for user behavior events
├── static/
│   ├── script.js           # Frontend logic for search and event logging
│   └── style.css           # Basic styling
└── templates/
    └── index.html          # Main HTML page
```

## Quick Start 🚀

### One-Click Launch (Recommended)

**Easiest way:** Simply double-click the `start_services.bat` file to automatically start all services.

**PowerShell method:**
```powershell
.\start_services.ps1
```

### Launch Process

The script will automatically execute the following steps:
1. **Start Docker services** - Run `docker-compose up --build`
2. **Wait for services to be ready** - Wait 10 seconds for services to fully start
3. **Start ngrok tunnel** - Create public access tunnel

### Access URLs

After startup, you can access through the following addresses:
- **Local access**: http://localhost:5000
- **ngrok console**: http://localhost:4040 (view public URL)
- **Public access**: Check specific address in ngrok console

### Important Notes

1. **Ensure ngrok is authenticated** - If using for the first time, run:
   ```
   ngrok config add-authtoken YOUR_TOKEN
   ```
2. **Docker must be running** - Ensure Docker Desktop is started
3. **Port availability** - Ensure port 5000 is not occupied by other programs

### Stopping Services

To stop services, you can:
- Close the respective command line windows
- Or stop containers in Docker Desktop
- Or run `docker-compose down`

---

## Manual Setup and Installation

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- [Python 3.8+](https://www.python.org/downloads/)

### 1. Start Elasticsearch

First, start the Elasticsearch container using Docker Compose. This will create a single-node cluster and expose it on port 9200.

```bash
docker-compose up -d
```

### 2. Install Python Dependencies

Install the required Python libraries using pip.

```bash
pip install -r requirements.txt
```

### 3. Prepare the Data

Run the `data_preprocessor.py` script to download the MS MARCO dataset, process it, and create the `msmarco_docs.json` file. This file will be used for indexing into Elasticsearch.

```bash
python data_preprocessor.py
```

### 4. Run the Application

Start the Flask web server. This script will also automatically create the Elasticsearch index and bulk-index the documents from `msmarco_docs.json` on the first run.

```bash
python app.py
```

The search engine will be available at `http://127.0.0.1:5000`.

## How to Use

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Enter a search query in the search box and click "Search".
3.  The results will be displayed below.

### Experimental Modes

You can control the quality of the search results by appending a `mode` parameter to the search URL. This is intended for experimental purposes.

- **Optimal Mode (Default):** Returns the top 10 most relevant results as determined by the re-ranking model.
  `http://127.0.0.1:5000/search?q=your-query&mode=optimal`

- **Degraded Relevance Mode:** Returns a mix of 5 highly relevant results and 5 less relevant results, shuffled together. This simulates a SERP with some noise.
  `http://127.0.0.1:5000/search?q=your-query&mode=degraded_relevance`

## User Behavior Logging

The system logs user interactions to `events.log` in JSON format. Each log entry includes a `type`, `sessionId`, `timestamp`, and event-specific data.

### Troubleshooting

If you encounter issues:
1. Check if Docker Desktop is running
2. Verify that ngrok is properly installed and authenticated
3. Check if port 5000 is occupied by other programs
4. Review command line output for error messages

---

### Logged Events

- **`query_submitted`:** When a user submits a query.
- **`serp_impression`:** When a set of search results is displayed.
- **`result_clicked`:** When a user clicks on a search result.
- **`query_abandoned`:** When a user issues a new query without clicking on any results from the previous one.

### Example Log Entry

```json
{"type": "result_clicked", "sessionId": "lq5o6x...","timestamp": "2023-10-27T10:00:00.123Z", "query": "what is python", "docId": "msmarco_doc_0_1", "rank": 1}
```

---

# <a name="中文"></a>中文

# 用于用户行为分析的现代搜索引擎

本项目是一个现代搜索引擎，旨在研究不同类型的非相关搜索结果如何影响用户行为和搜索效率。通过操控搜索结果的相关性和一致性，本系统可以分析用户互动模式，例如搜索耗时和查询放弃率。

## 主要功能

- **两阶段搜索:** 采用“召回 + 重排”架构，使用 Elasticsearch 进行初步候选结果检索，并使用 `sentence-transformer` 交叉编码器模型进行精细化重排序。
- **用户行为追踪:** 将关键的用户交互（包括查询、结果点击和查询放弃）记录到本地文件 (`events.log`) 中。
- **实验框架:** 支持 A/B 测试，可以提供不同“模式”的搜索结果，例如最优模式和“相关性降低”模式。
- **容器化环境:** 使用 Docker 管理 Elasticsearch 服务，确保环境的一致性和可复现性。

## 技术栈

- **后端:** Flask
- **搜索:** Elasticsearch 8.x
- **重排序模型:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **前端:** 原生 JavaScript, HTML, CSS
- **环境:** Docker, Python 3

## 项目结构

```
.SearchEngine/
├── app.py                  # Flask 主应用，包含搜索和日志记录的端点
├── data_preprocessor.py    # 用于下载和处理 MS MARCO 数据集的脚本
├── docker-compose.yml      # Elasticsearch 的 Docker 配置
├── requirements.txt        # Python 依赖
├── msmarco_docs.json       # 用于索引的已处理文档
├── events.log              # 用户行为事件的日志文件
├── static/
│   ├── script.js           # 用于搜索和事件记录的前端逻辑
│   └── style.css           # 基本样式
└── templates/
    └── index.html          # 主 HTML 页面
```

## 快速启动 🚀

### 一键启动（推荐）

**最简单方式：** 直接双击 `start_services.bat` 文件即可自动启动所有服务。

**PowerShell 方式：**
```powershell
.\start_services.ps1
```

### 启动流程

脚本会自动执行以下步骤：
1. **启动 Docker 服务** - 运行 `docker-compose up --build`
2. **等待服务就绪** - 等待 10 秒让服务完全启动
3. **启动 ngrok 隧道** - 创建公网访问隧道

### 访问地址

启动完成后，你可以通过以下地址访问：
- **本地访问**: http://localhost:5000
- **ngrok 控制台**: http://localhost:4040 （查看公网地址）
- **公网访问**: 在 ngrok 控制台中查看具体地址

### 注意事项

1. **确保 ngrok 已认证** - 如果是首次使用，需要先运行：
   ```
   ngrok config add-authtoken YOUR_TOKEN
   ```
2. **Docker 必须运行** - 确保 Docker Desktop 已启动
3. **端口占用** - 确保 5000 端口未被其他程序占用

### 停止服务

要停止服务，可以：
- 关闭相应的命令行窗口
- 或在 Docker Desktop 中停止容器
- 或运行 `docker-compose down`

---

## 手动安装与运行

### 环境要求

- [Docker](https://www.docker.com/get-started) 和 Docker Compose
- [Python 3.8+](https://www.python.org/downloads/)

### 1. 启动 Elasticsearch

首先，使用 Docker Compose 启动 Elasticsearch 容器。这将创建一个单节点集群，并将其暴露在 9200 端口。

```bash
docker-compose up -d
```

### 2. 安装 Python 依赖

使用 pip 安装所需的 Python 库。

```bash
pip install -r requirements.txt
```

### 3. 准备数据

运行 `data_preprocessor.py` 脚本来下载 MS MARCO 数据集，对其进行处理，并创建 `msmarco_docs.json` 文件。该文件将用于在 Elasticsearch 中建立索引。

```bash
python data_preprocessor.py
```

### 4. 运行应用

启动 Flask Web 服务器。该脚本在首次运行时也会自动创建 Elasticsearch 索引，并从 `msmarco_docs.json` 文件中批量索引文档。

```bash
python app.py
```

搜索引擎将在 `http://127.0.0.1:5000` 上可用。

## 如何使用

1.  打开您的网络浏览器，访问 `http://127.0.0.1:5000`。
2.  在搜索框中输入查询词，然后点击“搜索”。
3.  结果将显示在下方。

### 实验模式

您可以通过在搜索 URL 后附加一个 `mode` 参数来控制搜索结果的质量。这主要用于实验目的。

- **最优模式 (默认):** 返回由重排序模型确定的前 10 个最相关的结果。
  `http://127.0.0.1:5000/search?q=your-query&mode=optimal`

- **相关性降低模式:** 返回 5 个高度相关和 5 个相关性较低的结果的混合，并打乱顺序。这可以模拟一个带有噪声的搜索结果页面。
  `http://127.0.0.1:5000/search?q=your-query&mode=degraded_relevance`

## 用户行为日志

系统以 JSON 格式将用户交互记录到 `events.log` 文件中。每个日志条目都包含 `type`、`sessionId`、`timestamp` 和特定于事件的数据。

### 故障排除

如果遇到问题：
1. 检查 Docker Desktop 是否正在运行
2. 检查 ngrok 是否已正确安装和认证
3. 检查端口 5000 是否被占用
4. 查看命令行输出的错误信息

---

### 记录的事件

- **`query_submitted`:** 用户提交查询时。
- **`serp_impression`:** 一组搜索结果显示时。
- **`result_clicked`:** 用户点击搜索结果时。
- **`query_abandoned`:** 用户在未点击前一次搜索的任何结果的情况下，发起了新的查询。

### 日志条目示例

```json
{"type": "result_clicked", "sessionId": "lq5o6x...","timestamp": "2023-10-27T10:00:00.123Z", "query": "what is python", "docId": "msmarco_doc_0_1", "rank": 1}
```