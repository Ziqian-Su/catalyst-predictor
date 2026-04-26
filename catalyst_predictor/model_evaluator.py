"""
模型评估模块
功能：多模型性能对比、可视化图表输出
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


def evaluate_model(model, X_test, y_test, X_train, y_train, cv=5):
    """
    评估单个模型的性能
    
    参数：
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集目标
        X_train: 训练集特征（用于CV）
        y_train: 训练集目标（用于CV）
        cv: 交叉验证折数
    
    返回：
        results: 包含各项指标的字典
    """
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    results = {
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_pred),
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results


def compare_models(models, X_test, X_test_s, y_test, X_train, X_train_s, y_train, cv=5):
    """
    多模型性能对比
    
    参数：
        models: 模型字典
        X_test: 测试集特征（原始值）
        X_test_s: 测试集特征（标准化）
        y_test: 测试集目标
        X_train: 训练集特征（原始值）
        X_train_s: 训练集特征（标准化）
        y_train: 训练集目标
        cv: 交叉验证折数
    
    返回：
        results_df: 性能对比DataFrame
    """
    results_list = []
    
    use_scaled = ['支持向量机', '神经网络']
    
    for name, model in models.items():
        if name in use_scaled:
            results = evaluate_model(model, X_test_s, y_test, X_train_s, y_train, cv)
        else:
            results = evaluate_model(model, X_test, y_test, X_train, y_train, cv)
        results['Model'] = name
        results_list.append(results)
    
    results_df = pd.DataFrame(results_list)
    # 按测试集R²降序排列
    results_df = results_df[['Model', 'Train_R2', 'Test_R2', 'CV_R2_mean', 
                               'CV_R2_std', 'RMSE', 'MAE']]
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    return results_df


def print_comparison_table(results_df, title="各模型性能对比"):
    """
    打印性能对比表到控制台
    
    参数：
        results_df: compare_models返回的DataFrame
        title: 表格标题
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'模型':<12} {'测试集R²':<10} {'测试集MAE':<10} {'5折CV R²(均值±标准差)'}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<12} {row['Test_R2']:<10.3f} {row['MAE']:<10.3f} "
              f"{row['CV_R2_mean']:.3f} ± {row['CV_R2_std']:.3f}")
    print("=" * 70)
    
    best = results_df.iloc[0]
    print(f"\n最佳模型: {best['Model']} (R²={best['Test_R2']:.3f}, MAE={best['MAE']:.3f})")
    
    # 过拟合检查
    print("\n过拟合检查 (训练R² - 测试R²):")
    for _, row in results_df.iterrows():
        gap = row['Train_R2'] - row['Test_R2']
        status = "正常" if gap <= 0.08 else "可能过拟合"
        print(f"  {row['Model']:<12} 差距={gap:.3f}  {status}")


def plot_scatter_grid(models, X_test, X_test_s, y_test, save_path, filename='fig_base_models_scatter.png'):
    """
    生成五模型散点图组（2×3布局）
    
    参数：
        models: 模型字典
        X_test: 测试集特征（原始值）
        X_test_s: 测试集特征（标准化）
        y_test: 测试集目标
        save_path: 保存目录
        filename: 文件名
    """
    # 创建0-100缩放器
    scaler_100 = MinMaxScaler(feature_range=(0, 100))
    y_test_scaled = scaler_100.fit_transform(y_test.values.reshape(-1, 1)).flatten()
    
    use_scaled = ['支持向量机', '神经网络']
    model_names = list(models.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, name in enumerate(model_names):
        model = models[name]
        X_use = X_test_s if name in use_scaled else X_test
        y_pred = model.predict(X_use)
        y_pred_scaled = scaler_100.transform(y_pred.reshape(-1, 1)).flatten()
        
        r2 = r2_score(y_test, y_pred)
        
        ax = axes[idx]
        ax.scatter(y_test_scaled, y_pred_scaled, alpha=1.0, s=15, c='blue', edgecolors='none')
        ax.plot([0, 100], [0, 100], 'r--', lw=1.5, alpha=0.8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('真实值 (%)', fontsize=10)
        ax.set_ylabel('预测值 (%)', fontsize=10)
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.text(5, 95, f'$R^2$ = {r2:.3f}', fontsize=9, verticalalignment='top')
    
    # 隐藏第6个子图
    axes[5].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")


def plot_performance_bar(results_df, save_path, filename='fig_performance_comparison.png'):
    """
    生成性能对比柱状图（分组柱状图：R² + MAE）
    
    参数：
        results_df: 包含性能指标的DataFrame
        save_path: 保存目录
        filename: 文件名
    """
    models = results_df['Model'].tolist()
    test_r2 = results_df['Test_R2'].tolist()
    mae = results_df['MAE'].tolist()
    
    # MAE缩放因子
    mae_scaling = 5
    mae_scaled = [m * mae_scaling for m in mae]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # R²柱
    bars1 = ax.bar(x - width/2, test_r2, width, color='#2E86AB', 
                   edgecolor='white', linewidth=0.5, label='测试集$R^2$')
    
    # MAE柱
    bars2 = ax.bar(x + width/2, mae_scaled, width, color='#E67E22', 
                   edgecolor='white', linewidth=0.5, label=f'测试集MAE (×{mae_scaling})')
    
    # 标注R²值
    for bar, val in zip(bars1, test_r2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#2E86AB')
    
    # 标注MAE值（原始值）
    for bar, val in zip(bars2, mae):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#E67E22')
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('$R^2$ / MAE (×5)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('各模型性能对比', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")