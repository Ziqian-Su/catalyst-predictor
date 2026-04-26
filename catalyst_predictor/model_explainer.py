"""
模型解释性分析模块
功能：SHAP值计算、特征重要性分析、SHAP可视化
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr


def compute_shap_values(xgb_model, X_train, X_test, cache_dir=None):
    """
    计算SHAP值（支持缓存）
    
    参数：
        xgb_model: 训练好的XGBoost模型
        X_train: 训练集特征
        X_test: 测试集特征
        cache_dir: 缓存目录
    
    返回：
        shap_values: SHAP值矩阵
    """
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'shap_values.npy')
        if os.path.exists(cache_file):
            print("发现SHAP缓存，直接加载...")
            return np.load(cache_file)
    
    print("计算SHAP值（首次运行约2分钟）...")
    
    # 创建背景数据
    background = shap.kmeans(X_train, 50)
    
    # 包装函数避免兼容性问题
    def predict_func(X):
        if isinstance(X, pd.DataFrame):
            return xgb_model.predict(X)
        else:
            return xgb_model.predict(pd.DataFrame(X, columns=X_train.columns))
    
    # 创建解释器
    explainer = shap.KernelExplainer(predict_func, background.data)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)
    
    # 保存缓存
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_file, shap_values)
        print(f"SHAP缓存已保存至: {cache_file}")
    
    return shap_values


def get_feature_importance(xgb_model, features):
    """
    获取Gain特征重要性
    
    返回：
        importance_df: 特征重要性DataFrame
    """
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def get_shap_importance(shap_values, features):
    """
    获取SHAP特征重要性
    
    返回：
        shap_imp_df: SHAP重要性DataFrame
    """
    shap_imp = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        'feature': features,
        'shap_importance': shap_imp
    }).sort_values('shap_importance', ascending=False)
    
    return shap_imp_df


def compare_importance_methods(gain_df, shap_df):
    """
    对比Gain和SHAP两种重要性评估方法
    
    返回：
        comparison: 对比结果字典
    """
    gain_top10 = set(gain_df.head(10)['feature'])
    shap_top10 = set(shap_df.head(10)['feature'])
    common = gain_top10 & shap_top10
    
    # Spearman相关性
    common_all = list(set(gain_df.head(20)['feature']) & set(shap_df.head(20)['feature']))
    gain_rank = {f: i+1 for i, f in enumerate(gain_df['feature'])}
    shap_rank = {f: i+1 for i, f in enumerate(shap_df['feature'])}
    corr, pval = spearmanr([gain_rank[f] for f in common_all], 
                           [shap_rank[f] for f in common_all])
    
    return {
        'common_count': len(common),
        'common_features': common,
        'spearman_corr': corr,
        'spearman_pval': pval
    }


def plot_importance_bar(gain_df, save_path, top_n=15, filename='fig_xgboost_importance.png'):
    """
    特征重要性柱状图（垂直）
    """
    top_features = gain_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(top_features))
    bars = ax.bar(x, top_features['importance'].values, color='#2E86AB', 
                  edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_features['feature'].values, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('特征重要性 (Gain)', fontsize=12)
    ax.set_xlabel('特征名称', fontsize=12)
    ax.set_title(f'XGBoost特征重要性（Top {top_n}）', fontsize=14, fontweight='bold', pad=20)
    ax.grid(False)
    
    for bar, val in zip(bars, top_features['importance'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, top_features['importance'].max() * 1.15)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")


def plot_shap_summary(shap_values, X_test, features, save_path, 
                      max_display=15, filename='fig_shap_summary.png'):
    """
    SHAP摘要图
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=features, 
                      max_display=max_display, show=False)
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.grid(False)
    plt.title('XGBoost SHAP摘要图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")


def plot_shap_dependence(shap_values, X_test, features, top_features, save_path,
                         filename='fig_shap_dependence.png'):
    """
    核心特征SHAP依赖图
    """
    top3 = top_features[:3]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(top3):
        feat_idx = features.index(feat)
        shap.dependence_plot(feat_idx, shap_values, X_test, feature_names=features,
                             show=False, ax=axes[idx])
        axes[idx].grid(False)
        axes[idx].set_xlabel(feat, fontsize=11)
        axes[idx].set_ylabel('SHAP值', fontsize=11)
        axes[idx].set_title(f'{feat}', fontweight='bold', fontsize=12)
    
    axes[3].set_visible(False)
    plt.suptitle('核心特征SHAP依赖图', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")


def print_importance_tables(gain_df, shap_df, top_n=10):
    """
    打印特征重要性表
    """
    print("\n" + "=" * 60)
    print(f"Gain特征重要性（Top {top_n}）")
    print("-" * 40)
    for i, (_, row) in enumerate(gain_df.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<20} {row['importance']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"SHAP特征重要性（Top {top_n}）")
    print("-" * 40)
    for i, (_, row) in enumerate(shap_df.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<20} {row['shap_importance']:.4f}")


def print_comparison(comparison):
    """
    打印重要性方法对比结果
    """
    print("\n" + "=" * 60)
    print("Gain vs SHAP 对比")
    print("-" * 40)
    print(f"  Top 10 共同特征: {comparison['common_count']} 个")
    print(f"  Spearman相关系数: {comparison['spearman_corr']:.3f} (p={comparison['spearman_pval']:.4f})")