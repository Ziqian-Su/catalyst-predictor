"""
Stacking集成学习模块
功能：构建Stacking集成模型、评估、权重分析、可视化
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


def build_stacking(models, X_train, y_train, random_state=42):
    """
    构建Stacking集成模型
    
    参数：
        models: 基模型字典（需包含XGBoost、随机森林、SVR、MLP）
        X_train: 标准化后的训练集特征
        y_train: 训练集目标
        random_state: 随机种子
    
    返回：
        stacking_model: 训练好的Stacking模型
    """
    base_models = [
        ('xgb', models['XGBoost']),
        ('rf', models['随机森林']),
        ('svr', models['支持向量机']),
        ('mlp', models['神经网络'])
    ]
    
    meta_model = Ridge(alpha=1.0, random_state=random_state)
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=False
    )
    
    print("训练Stacking集成模型...")
    stacking_model.fit(X_train, y_train)
    print("Stacking训练完成")
    
    return stacking_model


def evaluate_stacking(stacking_model, X_test, y_test, X_train, y_train, cv=5):
    """
    评估Stacking模型
    
    返回：
        results: 包含各项指标的字典
    """
    y_pred = stacking_model.predict(X_test)
    y_train_pred = stacking_model.predict(X_train)
    
    cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='r2')
    
    results = {
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_pred),
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results


def get_stacking_weights(stacking_model):
    """
    获取各基模型的权重系数
    
    返回：
        weights_df: 权重DataFrame
    """
    meta_model = stacking_model.final_estimator_
    base_names = ['XGBoost', '随机森林', '神经网络', '支持向量机']
    
    weights_df = pd.DataFrame({
        '基模型': base_names,
        '权重系数': meta_model.coef_
    }).sort_values('权重系数', ascending=False)
    
    weights_df['贡献度(%)'] = weights_df['权重系数'] / weights_df['权重系数'].sum() * 100
    
    return weights_df


def print_stacking_results(results, weights_df):
    """
    打印Stacking结果到控制台
    """
    print("\n" + "=" * 60)
    print("Stacking集成结果")
    print("=" * 60)
    print(f"  测试集R²: {results['Test_R2']:.3f}")
    print(f"  测试集MAE: {results['MAE']:.3f}")
    print(f"  5折CV R²: {results['CV_R2_mean']:.3f} ± {results['CV_R2_std']:.3f}")
    
    print("\n基模型权重分配:")
    print("-" * 50)
    print(f"{'基模型':<12} {'权重系数':<10} {'贡献度':<10}")
    print("-" * 50)
    for _, row in weights_df.iterrows():
        print(f"{row['基模型']:<12} {row['权重系数']:<10.4f} {row['贡献度(%)']:<10.1f}%")
    print("-" * 50)


def plot_stacking_scatter(stacking_model, X_test, y_test, save_path, 
                          filename='fig_stacking_scatter.png'):
    """
    生成Stacking散点图
    
    参数：
        stacking_model: 训练好的Stacking模型
        X_test: 标准化后的测试集特征
        y_test: 测试集目标
        save_path: 保存目录
        filename: 文件名
    """
    scaler_100 = MinMaxScaler(feature_range=(0, 100))
    y_test_scaled = scaler_100.fit_transform(y_test.values.reshape(-1, 1)).flatten()
    
    y_pred = stacking_model.predict(X_test)
    y_pred_scaled = scaler_100.transform(y_pred.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_scaled, y_pred_scaled, alpha=1.0, s=15, c='blue', edgecolors='none')
    plt.plot([0, 100], [0, 100], 'r--', lw=1.5, alpha=0.8)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal')
    plt.grid(False)
    plt.xlabel('真实值 (%)', fontsize=12)
    plt.ylabel('预测值 (%)', fontsize=12)
    plt.title('Stacking集成', fontweight='bold', fontsize=14)
    
    plt.text(5, 95, f'$R^2$ = {r2:.3f}', fontsize=11, verticalalignment='top')
    plt.text(5, 88, f'MAE = {mae:.3f}', fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")