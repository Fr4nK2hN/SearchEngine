# SearchEngine

基于 MS MARCO 语料的多阶段检索与重排序系统，包含：

- `webapp/`：Flask Web 应用、运行时装配、路由与服务层
- `engine/`：检索与索引内核
- `ranking/`：LTR、Router、评估与训练数据生成
- `pipelines/`：离线数据下载、预处理与训练入口
- `tools/`：实验、分析与报告生成脚本
- `docs/`：项目文档与阶段性说明

常用入口：

- `python app.py`
- `python init.py`
- `python wait_for_index.py`
- `python download_resources.py`
- `python data_preprocessor.py`
- `python download_msmarco.py`
- `python train_ltr_model.py`
