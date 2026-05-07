"""
完整建模流程运行脚本
使用方法：python examples/run_full_pipeline.py

用户只需修改 DATA_PATH、TARGET_COL、ID_COL 即可适配新数据集
注意：本代码仅开源建模流程，不包含数据集
"""
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from catalyst_predictor.data_loader import load_data, split_data
from catalyst_predictor.feature_selector import robustness_validation, get_selected_features
from catalyst_predictor.model_trainer import train_all_models
from catalyst_predictor.model_evaluator import compare_models, print_comparison_table, plot_scatter_grid, plot_performance_bar
from catalyst_predictor.stacking_trainer import build_stacking, evaluate_stacking, get_stacking_weights, print_stacking_results, plot_stacking_scatter
from catalyst_predictor.model_explainer import (
    compute_shap_values, get_feature_importance, get_shap_importance,
    compare_importance_methods, plot_importance_bar, plot_shap_summary,
    plot_shap_dependence, print_importance_tables, print_comparison
)
from catalyst_predictor.utils import ensure_dir, save_csv
import warnings
warnings.filterwarnings('ignore')

# ==================== 用户配置（修改这里适配新数据集） ====================
DATA_PATH = r'请在此填写您的数据文件路径'
TARGET_COL = 'targets'     # 目标变量列名
ID_COL = 'COFID'           # ID列名（如果没有则设为None）

# ==================== 参数配置 ====================
CONFIG = {
    # 特征筛选阶段XGBoost参数
    'xgb_params': {
        'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'importance_type': 'gain', 'random_state': 42, 'n_jobs': -1
    },
    # 特征筛选阈值
    'importance_cumulative_threshold': 0.95,
    'rank_std_threshold': 50,
    'consistency_threshold': 0.8,
    'correlation_threshold': 0.85,
    'cv_folds': 5,
    'n_iterations': 50,
    'test_size': 0.2,
    'random_state': 42,
    'random_seed_base': 42,
    # XGBoost超参数搜索配置
    'xgb_n_iter': 60,
    'xgb_search_space': {
        'n_estimators': (300, 500),
        'max_depth': (6, 10),
        'learning_rate': (0.03, 0.08),
        'subsample': (0.75, 0.2),
        'colsample_bytree': (0.75, 0.2),
        'gamma': (0, 0.1),
        'reg_alpha': (0.05, 0.2),
        'reg_lambda': (0.5, 0.8),
    },
    # 随机森林超参数搜索配置
    'rf_n_iter': 40,
    'rf_search_space': {
        'n_estimators': (80, 150),
        'max_depth': (4, 8),
        'min_samples_split': (5, 10),
        'min_samples_leaf': (3, 6),
        'max_features': ['sqrt', 0.4],
    },
    # SVR超参数搜索配置
    'svr_n_iter': 50,
    'svr_search_space': {
        'C': (0.5, 80),
        'gamma': (0.01, 0.15),
        'epsilon': (0.05, 0.1),
    },
    # MLP超参数搜索配置
    'mlp_n_iter': 30,
    'mlp_search_space': {
        'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
        'alpha': (0.01, 0.1),
        'learning_rate_init': (0.001, 0.01),
        'batch_size': [32, 64],
    },
}

RANDOM_STATE = 42
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# ==================== 创建输出目录 ====================
for d in [FIGURE_DIR, TABLE_DIR, MODEL_DIR]:
    ensure_dir(d)

print("=" * 60)
print("催化剂性能预测建模流程")
print("=" * 60)

# ==================== 1. 加载数据 ====================
print("\n[1/6] 加载数据...")
X, y, original_features, name_mapping = load_data(DATA_PATH, target_col=TARGET_COL, id_col=ID_COL)

# ==================== 2. 特征筛选 ====================
print("\n[2/6] 特征筛选...")
freq_df, _ = robustness_validation(X, y, CONFIG, cache_dir=MODEL_DIR)
selected_features = get_selected_features(freq_df, min_frequency=0.5)
print(f"选中 {len(selected_features)} 个特征")
save_csv(freq_df, os.path.join(TABLE_DIR, 'feature_frequency.csv'))

# ==================== 3. 划分数据 ====================
X_sel = X[selected_features]
X_train, X_test, y_train, y_test = split_data(X_sel, y, test_size=0.2, random_state=RANDOM_STATE)

# ==================== 4. 模型训练 ====================
print("\n[3/6] 模型训练...")
models, scaler = train_all_models(X_train, y_train, random_state=RANDOM_STATE, cache_dir=MODEL_DIR, config=CONFIG)

# ==================== 5. 模型评估 ====================
print("\n[4/6] 模型评估...")
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

results_df = compare_models(models, X_test, X_test_s, y_test, X_train, X_train_s, y_train, cv=5)
print_comparison_table(results_df, "五种基模型性能对比")
save_csv(results_df, os.path.join(TABLE_DIR, 'model_comparison_results.csv'))

plot_scatter_grid(models, X_test, X_test_s, y_test, FIGURE_DIR)
plot_performance_bar(results_df, FIGURE_DIR)

# ==================== 6. Stacking集成 ====================
print("\n[5/6] Stacking集成...")
stacking_model = build_stacking(models, X_train_s, y_train, random_state=RANDOM_STATE)
stacking_results = evaluate_stacking(stacking_model, X_test_s, y_test, X_train_s, y_train)
weights_df = get_stacking_weights(stacking_model)
print_stacking_results(stacking_results, weights_df)
save_csv(weights_df, os.path.join(TABLE_DIR, 'stacking_weights.csv'))
plot_stacking_scatter(stacking_model, X_test_s, y_test, FIGURE_DIR)

# ==================== 7. 模型解释 ====================
print("\n[6/6] 模型解释...")
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"自动选择最优模型进行解释: {best_model_name}")

gain_df = get_feature_importance(best_model, selected_features)
shap_values = compute_shap_values(best_model, X_train, X_test, cache_dir=MODEL_DIR)
shap_df = get_shap_importance(shap_values, selected_features)

print_importance_tables(gain_df, shap_df)
comparison = compare_importance_methods(gain_df, shap_df)
print_comparison(comparison)

save_csv(gain_df, os.path.join(TABLE_DIR, 'feature_importance_gain.csv'))
save_csv(shap_df, os.path.join(TABLE_DIR, 'feature_importance_shap.csv'))

plot_importance_bar(gain_df, FIGURE_DIR)
plot_shap_summary(shap_values, X_test, selected_features, FIGURE_DIR)
plot_shap_dependence(shap_values, X_test, selected_features, gain_df['feature'].tolist(), FIGURE_DIR)

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("完成！所有输出已保存至 outputs/")
print(f"  图片: {FIGURE_DIR}")
print(f"  表格: {TABLE_DIR}")
print(f"  模型: {MODEL_DIR}")
print("=" * 60)