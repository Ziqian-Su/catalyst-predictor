"""
特征筛选模块
实现三步法特征筛选：重要性筛选 → 稳定性筛选 → 冗余性筛选
支持缓存加载，避免重复计算
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import pearsonr


def three_step_selection(X_train, y_train, config):
    """
    执行一次完整的三步法特征筛选
    
    参数：
        X_train: 训练集特征DataFrame
        y_train: 训练集目标变量
        config: 配置字典，需包含：
            - xgb_params: XGBoost参数
            - importance_cumulative_threshold: 累计重要性阈值
            - rank_std_threshold: 排名标准差阈值
            - consistency_threshold: 一致性阈值
            - correlation_threshold: 相关系数阈值
            - cv_folds: 交叉验证折数
            - random_seed_base: 随机种子
    
    返回：
        selected_features: 最终筛选出的特征名列表
        metrics_df: 各特征的评估指标DataFrame
    """
    features = X_train.columns.tolist()
    n_features = len(features)
    cv_folds = config.get('cv_folds', 5)
    random_seed = config.get('random_seed_base', config.get('random_state', 42))
    xgb_params = config.get('xgb_params', {})
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    
    # 存储各折的重要性得分和排名
    importance_data = np.zeros((n_features, cv_folds))
    rank_data = np.zeros((n_features, cv_folds))
    
    # 5折交叉验证
    for fold, (train_idx, _) in enumerate(kf.split(X_train)):
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_importance = model.feature_importances_
        importance_data[:, fold] = fold_importance
        fold_rank = pd.Series(fold_importance).rank(ascending=False, method='dense').values
        rank_data[:, fold] = fold_rank
    
    # 计算评估指标
    metrics_df = pd.DataFrame({
        'feature': features,
        'importance_mean': importance_data.mean(axis=1),
        'importance_std': importance_data.std(axis=1),
        'rank_mean': rank_data.mean(axis=1),
        'rank_std': rank_data.std(axis=1),
        'consistency': (importance_data > 0).sum(axis=1) / cv_folds
    }).set_index('feature')
    
    # ========== 第一步：重要性筛选 ==========
    sorted_features = metrics_df.sort_values('importance_mean', ascending=False)
    sorted_features['cumulative_importance'] = (
        sorted_features['importance_mean'].cumsum() / sorted_features['importance_mean'].sum()
    )
    
    threshold = config.get('importance_cumulative_threshold', 0.95)
    step1_features = sorted_features[
        sorted_features['cumulative_importance'] <= threshold
    ].index.tolist()
    
    if not step1_features:
        step1_features = sorted_features.head(10).index.tolist()
    
    # ========== 第二步：稳定性筛选 ==========
    step1_metrics = metrics_df.loc[step1_features]
    rank_threshold = config.get('rank_std_threshold', 50)
    consistency_threshold = config.get('consistency_threshold', 0.8)
    
    step2_mask = (
        (step1_metrics['rank_std'] <= rank_threshold) & 
        (step1_metrics['consistency'] >= consistency_threshold)
    )
    step2_features = step1_metrics[step2_mask].index.tolist()
    
    if not step2_features:
        step2_features = step1_features
    
    # ========== 第三步：冗余性筛选 ==========
    if len(step2_features) > 1:
        step2_metrics = metrics_df.loc[step2_features].sort_values(
            'importance_mean', ascending=False
        )
        step2_features_sorted = step2_metrics.index.tolist()
        
        # 计算各特征与目标变量的相关性
        target_corr = {}
        for feat in step2_features_sorted:
            corr, _ = pearsonr(X_train[feat], y_train)
            target_corr[feat] = abs(corr)
        
        corr_threshold = config.get('correlation_threshold', 0.85)
        selected = []
        remaining = step2_features_sorted.copy()
        
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            if not remaining:
                break
            # 计算当前特征与剩余特征的相关性
            corr_with_current = X_train[remaining].apply(
                lambda col: abs(pearsonr(X_train[current], col)[0])
            )
            high_corr = corr_with_current[corr_with_current > corr_threshold].index.tolist()
            for corr_feat in high_corr:
                if target_corr.get(corr_feat, 0) <= target_corr.get(current, 0):
                    if corr_feat in remaining:
                        remaining.remove(corr_feat)
        final_features = selected
    else:
        final_features = step2_features
    
    return final_features, metrics_df


def robustness_validation(X, y, config, cache_dir=None):
    """
    50次稳健性验证，统计特征入选频率（支持缓存）
    
    参数：
        X: 全量特征DataFrame
        y: 目标变量
        config: 配置字典
        cache_dir: 缓存目录路径，如果为None则不缓存
    
    返回：
        frequency_df: 特征入选频率表
        all_selected: 每次迭代选中的特征列表
    """
    # 检查缓存
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'feature_frequency.pkl')
        if os.path.exists(cache_file):
            print("发现特征筛选缓存，直接加载...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"已加载 {len(cache_data['frequency_df'])} 个特征的频率数据")
            return cache_data['frequency_df'], cache_data['all_selected']
    
    n_iterations = config.get('n_iterations', 50)
    test_size = config.get('test_size', 0.2)
    random_seed_base = config.get('random_state', 42)
    
    all_selected = []
    
    print(f"开始稳健性验证：{n_iterations}次随机划分测试")
    
    for i in range(n_iterations):
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{n_iterations} 次")
        
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, 
            random_state=random_seed_base + i
        )
        
        selected_features, _ = three_step_selection(X_train, y_train, config)
        all_selected.append(selected_features)
    
    print(f"稳健性验证完成。")
    
    # 统计入选频率
    all_features_flat = [feat for sublist in all_selected for feat in sublist]
    feature_counts = pd.Series(all_features_flat).value_counts()
    
    frequency_df = pd.DataFrame({
        'feature': feature_counts.index,
        'count': feature_counts.values,
        'frequency': feature_counts.values / n_iterations
    }).sort_values('frequency', ascending=False)
    
    # 划分等级
    thresholds = config.get('feature_level_thresholds', {
        '核心(>90%)': 0.90,
        '稳定(80-90%)': 0.80,
        '常见(50-80%)': 0.50,
        '偶然(<50%)': 0.00
    })
    
    def classify_level(freq):
        for level_name, level_threshold in thresholds.items():
            if freq >= level_threshold:
                return level_name
        return '未选中'
    
    frequency_df['level'] = frequency_df['frequency'].apply(classify_level)
    
    # 保存缓存
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'frequency_df': frequency_df, 'all_selected': all_selected}, f)
        print(f"特征筛选缓存已保存至: {cache_file}")
    
    return frequency_df, all_selected


def get_selected_features(frequency_df, min_frequency=0.5):
    """
    根据频率阈值获取最终选中的特征列表
    
    参数：
        frequency_df: 稳健性验证的频率表
        min_frequency: 最低入选频率（默认0.5，即50%）
    
    返回：
        selected_features: 特征名列表
    """
    selected = frequency_df[frequency_df['frequency'] >= min_frequency]
    return selected['feature'].tolist()