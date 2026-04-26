"""
模型训练模块
功能：五种基模型的超参数优化与训练
支持缓存加载，避免重复训练
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import uniform, randint


def optimize_xgboost(X_train, y_train, random_state=42, n_iter=60):
    """
    XGBoost超参数优化
    
    参数：
        X_train: 训练集特征
        y_train: 训练集目标
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
    
    返回：
        best_model: 最优XGBoost模型
    """
    param_dist = {
        'n_estimators': randint(300, 500),
        'max_depth': randint(6, 10),
        'learning_rate': uniform(0.03, 0.08),
        'subsample': uniform(0.75, 0.2),
        'colsample_bytree': uniform(0.75, 0.2),
        'gamma': uniform(0, 0.1),
        'min_child_weight': randint(2, 5),
        'reg_alpha': uniform(0.05, 0.2),
        'reg_lambda': uniform(0.5, 0.8)
    }
    
    model = xgb.XGBRegressor(
        random_state=random_state, 
        n_jobs=-1, 
        importance_type='gain'
    )
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5, 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  XGBoost: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_random_forest(X_train, y_train, random_state=42, n_iter=40):
    """
    随机森林超参数优化
    
    参数：
        X_train: 训练集特征
        y_train: 训练集目标
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
    
    返回：
        best_model: 最优随机森林模型
    """
    param_dist = {
        'n_estimators': randint(80, 150),
        'max_depth': randint(4, 8),
        'min_samples_split': randint(5, 10),
        'min_samples_leaf': randint(3, 6),
        'max_features': ['sqrt', 0.4]
    }
    
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5, 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  随机森林: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_svr(X_train, y_train, random_state=42, n_iter=50):
    """
    支持向量机超参数优化
    
    参数：
        X_train: 标准化后的训练集特征
        y_train: 训练集目标
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
    
    返回：
        best_model: 最优SVR模型
    """
    param_dist = {
        'C': uniform(0.5, 80),
        'gamma': uniform(0.01, 0.15),
        'epsilon': uniform(0.05, 0.1)
    }
    
    model = SVR(kernel='rbf')
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5, 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  SVR: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_ridge(X_train, y_train, random_state=42):
    """
    岭回归超参数优化（网格搜索）
    
    参数：
        X_train: 训练集特征
        y_train: 训练集目标
        random_state: 随机种子
    
    返回：
        best_model: 最优岭回归模型
    """
    alphas = np.logspace(-2, 1, 30)
    
    model = Ridge(random_state=random_state)
    
    search = GridSearchCV(
        model, {'alpha': alphas}, cv=5, 
        scoring='r2', n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  岭回归: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_mlp(X_train, y_train, random_state=42, n_iter=30):
    """
    多层感知机超参数优化
    
    参数：
        X_train: 标准化后的训练集特征
        y_train: 训练集目标
        random_state: 随机种子
        n_iter: 随机搜索迭代次数
    
    返回：
        best_model: 最优MLP模型
    """
    param_dist = {
        'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
        'alpha': uniform(0.01, 0.1),
        'learning_rate_init': uniform(0.001, 0.01),
        'batch_size': [32, 64],
    }
    
    model = MLPRegressor(
        random_state=random_state,
        max_iter=1000,
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
    )
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5, 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  MLP: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def train_all_models(X_train, y_train, random_state=42, cache_dir=None):
    """
    训练所有五种基模型，支持缓存加载
    
    参数：
        X_train: 训练集特征（原始值）
        y_train: 训练集目标
        random_state: 随机种子
        cache_dir: 缓存目录路径，如果为None则不缓存
    
    返回：
        models: 模型字典
        scaler: 标准化器
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # 检查缓存
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'trained_models.pkl')
        scaler_file = os.path.join(cache_dir, 'scaler.pkl')
        if os.path.exists(cache_file) and os.path.exists(scaler_file):
            print("=" * 50)
            print("发现已训练模型缓存，直接加载...")
            with open(cache_file, 'rb') as f:
                models = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            print(f"已加载 {len(models)} 个模型")
            print("=" * 50)
            return models, scaler
    
    print("=" * 50)
    print("未发现缓存，开始训练模型...")
    print("=" * 50)
    
    print("\n[1/5] XGBoost...")
    xgb_model = optimize_xgboost(X_train, y_train, random_state)
    
    print("\n[2/5] 随机森林...")
    rf_model = optimize_random_forest(X_train, y_train, random_state)
    
    print("\n[3/5] 支持向量机（标准化数据）...")
    svr_model = optimize_svr(X_train_s, y_train, random_state)
    
    print("\n[4/5] 岭回归...")
    ridge_model = optimize_ridge(X_train, y_train, random_state)
    
    print("\n[5/5] 神经网络（标准化数据）...")
    mlp_model = optimize_mlp(X_train_s, y_train, random_state)
    
    models = {
        'XGBoost': xgb_model,
        '随机森林': rf_model,
        '支持向量机': svr_model,
        '岭回归': ridge_model,
        '神经网络': mlp_model
    }
    
    # 保存缓存
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'trained_models.pkl'), 'wb') as f:
            pickle.dump(models, f)
        with open(os.path.join(cache_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n模型缓存已保存至: {cache_dir}")
    
    print("\n" + "=" * 50)
    print("所有模型训练完成！")
    print("=" * 50)
    
    return models, scaler