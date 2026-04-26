"""
默认配置文件
包含所有可调参数，修改此文件即可调整整个建模流程
"""
import os

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'activity_task-Data578-Feature1362.csv')

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# ==================== 数据划分配置 ====================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== 三步法特征筛选配置 ====================
N_ITERATIONS = 50
CV_FOLDS = 5
IMPORTANCE_CUMULATIVE_THRESHOLD = 0.95
RANK_STD_THRESHOLD = 50
CONSISTENCY_THRESHOLD = 0.8
CORRELATION_THRESHOLD = 0.85

# ==================== XGBoost默认参数 ====================
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'importance_type': 'gain',
    'random_state': 42,
    'n_jobs': -1
}

# ==================== 超参数搜索空间 ====================
XGB_SEARCH_SPACE = {
    'n_estimators': (300, 500),
    'max_depth': (6, 10),
    'learning_rate': (0.03, 0.08),
    'subsample': (0.75, 0.2),
    'colsample_bytree': (0.75, 0.2),
    'gamma': (0, 0.1),
    'reg_alpha': (0.05, 0.2),
    'reg_lambda': (0.5, 0.8)
}

RF_SEARCH_SPACE = {
    'n_estimators': (80, 150),
    'max_depth': (4, 8),
    'min_samples_split': (5, 10),
    'min_samples_leaf': (3, 6),
    'max_features': ['sqrt', 0.4]
}

SVR_SEARCH_SPACE = {
    'C': (0.5, 80),
    'gamma': (0.01, 0.15),
    'epsilon': (0.05, 0.1)
}

MLP_SEARCH_SPACE = {
    'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
    'alpha': (0.01, 0.1),
    'learning_rate_init': (0.001, 0.01),
    'batch_size': [32, 64]
}

# ==================== 解释性分析配置 ====================
SHAP_BACKGROUND_SAMPLES = 50
SHAP_TOP_FEATURES = 15
SHAP_DEPENDENCE_TOP_N = 3

# ==================== 图片保存配置 ====================
FIGURE_DPI = 300
FIGURE_COLOR = '#2E86AB'