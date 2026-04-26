# catalyst-predictor

基于机器学习的电催化CO₂还原性能预测工具包

## 简介

本工具包是毕业论文《基于机器学习的电催化二氧化碳还原性能预测研究》的配套代码，
封装了从数据加载、特征筛选、模型训练、性能评估、Stacking集成到SHAP模型解释的
完整建模流程。

## 模块架构

```
catalyst_predictor/
├── config/
│   └── default_config.py          # 集中式参数配置
├── catalyst_predictor/
│   ├── __init__.py                # 包初始化
│   ├── data_loader.py             # 数据加载与特征名规范化
│   ├── feature_selector.py        # 三步法特征筛选
│   ├── model_trainer.py           # 模型训练与超参数优化
│   ├── model_evaluator.py         # 模型评估与可视化
│   ├── stacking_trainer.py        # Stacking集成学习
│   ├── model_explainer.py         # SHAP模型解释
│   └── utils.py                   # 通用工具函数
├── examples/
│   └── run_full_pipeline.py       # 全流程一键运行脚本
├── outputs/                       # 输出目录（自动创建）
│   ├── figures/                   # 可视化图表
│   ├── tables/                    # 结果表格
│   └── models/                    # 模型缓存
├── docs/                          # 使用文档
├── README.md                      # 项目说明
├── requirements.txt               # 依赖列表
└── LICENSE                        # 开源许可证
```
## 快速开始

### 环境要求

- Python 3.9 及以上
- 依赖包：numpy, pandas, scikit-learn, xgboost, matplotlib, shap, scipy

### 安装

git clone https://github.com/xxx/catalyst-predictor.git
cd catalyst-predictor
pip install -r requirements.txt

### 运行

1. 打开 examples/run_full_pipeline.py
2. 修改数据路径和目标列名：

DATA_PATH = r'您的数据文件路径.csv'
TARGET_COL = 'targets'     # 目标变量列名
ID_COL = 'COFID'           # ID列名（没有则设为None）

3. 运行：

python examples/run_full_pipeline.py

## 输出结果

所有输出自动保存至 outputs/ 目录：

### 可视化图表（figures/）

| 文件名 | 对应论文 | 说明 |
|--------|---------|------|
| fig_base_models_scatter.png | 图3-1 | 五模型预测散点图 |
| fig_performance_comparison.png | 图3-3 | 性能对比柱状图 |
| fig_stacking_scatter.png | 图3-2 | Stacking集成散点图 |
| fig_xgboost_importance.png | 图4-1 | 特征重要性柱状图 |
| fig_shap_summary.png | 图4-2 | SHAP摘要图 |
| fig_shap_dependence.png | 图4-3 | SHAP依赖图 |

### 结果表格（tables/）

| 文件名 | 说明 |
|--------|------|
| feature_frequency.csv | 特征入选频率表 |
| model_comparison_results.csv | 模型性能对比表 |
| stacking_weights.csv | Stacking基模型权重表 |
| feature_importance_gain.csv | Gain特征重要性排序 |
| feature_importance_shap.csv | SHAP特征重要性排序 |

### 模型缓存（models/）

| 文件名 | 说明 |
|--------|------|
| trained_models.pkl | 五个训练好的基模型 |
| stacking_model.pkl | Stacking集成模型 |
| scaler.pkl | 标准化器 |
| shap_values.npy | SHAP值缓存 |

## 模块说明

| 模块 | 功能 |
|------|------|
| data_loader.py | 数据加载、特征名清理、数据集划分 |
| feature_selector.py | 三步法特征筛选 + 50次稳健性验证 |
| model_trainer.py | 五模型超参数优化与训练 |
| model_evaluator.py | 多模型性能对比、散点图、柱状图 |
| stacking_trainer.py | Stacking集成训练、权重分析、散点图 |
| model_explainer.py | SHAP值计算、特征重要性、可视化 |

## 参数配置

### 基础参数（run_full_pipeline.py）

需要用户自行修改的三个参数：

DATA_PATH = r'新数据路径.csv'   # 数据文件路径
TARGET_COL = '新目标列名'       # 目标变量列名
ID_COL = '新ID列名'             # ID列名（没有则设为None）

### 高级参数（config/default_config.py）

所有建模参数集中在配置文件中，可按需修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TEST_SIZE | 0.2 | 测试集比例 |
| RANDOM_STATE | 42 | 随机种子 |
| N_ITERATIONS | 50 | 稳健性验证迭代次数 |
| CV_FOLDS | 5 | 交叉验证折数 |
| IMPORTANCE_CUMULATIVE_THRESHOLD | 0.95 | 重要性累计贡献率阈值 |
| RANK_STD_THRESHOLD | 50 | 排名标准差阈值 |
| CONSISTENCY_THRESHOLD | 0.8 | 一致性阈值 |
| CORRELATION_THRESHOLD | 0.85 | 冗余特征相关系数阈值 |
| SHAP_BACKGROUND_SAMPLES | 50 | SHAP背景数据样本数 |

修改配置文件后保存，重新运行 run_full_pipeline.py 即可生效。

## 适配新数据集

只需修改 run_full_pipeline.py 中三个基础参数，其余代码无需改动，一键运行即可。

## 注意事项

- 本工具包不包含实验数据，用户需自行准备CSV格式数据集
- 数据需包含目标列，可选包含ID列（会自动排除）
- 特征名中的特殊字符（如 [ ] < >）会自动清理，无需手动处理
- 首次运行需5-10分钟，后续运行自动加载缓存，秒级完成
- 如遇字体警告，需确保系统已安装SimHei中文字体

## 引用

苏子骞. 基于机器学习的电催化二氧化碳还原性能预测研究[D].
哈尔滨工业大学深圳校区, 2026.

## 许可证

MIT License
