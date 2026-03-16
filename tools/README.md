# Tools

脚本按职责分层：

- `tools/analysis/`：性能分析、绘图、报告导出
- `tools/experiments/`：离线实验、ablation、router/oracle 评估
- `tools/maintenance/`：环境检查、数据构建、IDF/向量预计算

为兼容已有命令，`tools/*.py` 仍保留为薄入口文件。
