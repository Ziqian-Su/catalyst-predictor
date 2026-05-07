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


def optimize_xgboost(X_train, y_train, random_state=42, n_iter=60, search_space=None):
    if search_space is None:
        search_space = {
            'n_estimators': (300, 500), 'max_depth': (6, 10),
            'learning_rate': (0.03, 0.08), 'subsample': (0.75, 0.2),
            'colsample_bytree': (0.75, 0.2), 'gamma': (0, 0.1),
            'reg_alpha': (0.05, 0.2), 'reg_lambda': (0.5, 0.8),
        }
    
    param_dist = {
        'n_estimators': randint(*search_space['n_estimators']),
        'max_depth': randint(*search_space['max_depth']),
        'learning_rate': uniform(*search_space['learning_rate']),
        'subsample': uniform(*search_space['subsample']),
        'colsample_bytree': uniform(*search_space['colsample_bytree']),
        'gamma': uniform(*search_space['gamma']),
        'min_child_weight': randint(2, 5),
        'reg_alpha': uniform(*search_space['reg_alpha']),
        'reg_lambda': uniform(*search_space['reg_lambda']),
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


def optimize_random_forest(X_train, y_train, random_state=42, n_iter=40, search_space=None):
    if search_space is None:
        search_space = {
            'n_estimators': (80, 150), 'max_depth': (4, 8),
            'min_samples_split': (5, 10), 'min_samples_leaf': (3, 6),
            'max_features': ['sqrt', 0.4],
        }
    
    param_dist = {
        'n_estimators': randint(*search_space['n_estimators']),
        'max_depth': randint(*search_space['max_depth']),
        'min_samples_split': randint(*search_space['min_samples_split']),
        'min_samples_leaf': randint(*search_space['min_samples_leaf']),
        'max_features': search_space['max_features'],
    }
    
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5, 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  随机森林: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_svr(X_train, y_train, random_state=42, n_iter=50, search_space=None):
    if search_space is None:
        search_space = {
            'C': (0.5, 80), 'gamma': (0.01, 0.15), 'epsilon': (0.05, 0.1),
        }
    
    param_dist = {
        'C': uniform(*search_space['C']),
        'gamma': uniform(*search_space['gamma']),
        'epsilon': uniform(*search_space['epsilon']),
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
    alphas = np.logspace(-2, 1, 30)
    
    model = Ridge(random_state=random_state)
    
    search = GridSearchCV(
        model, {'alpha': alphas}, cv=5, 
        scoring='r2', n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    print(f"  岭回归: CV R² = {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_mlp(X_train, y_train, random_state=42, n_iter=30, search_space=None):
    if search_space is None:
        search_space = {
            'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
            'alpha': (0.01, 0.1),
            'learning_rate_init': (0.001, 0.01),
            'batch_size': [32, 64],
        }
    
    param_dist = {
        'hidden_layer_sizes': search_space['hidden_layer_sizes'],
        'alpha': uniform(*search_space['alpha']),
        'learning_rate_init': uniform(*search_space['learning_rate_init']),
        'batch_size': search_space['batch_size'],
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


def train_all_models(X_train, y_train, random_state=42, cache_dir=None, config=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
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
    
    if config:
        xgb_n_iter = config.get('xgb_n_iter', 60)
        rf_n_iter = config.get('rf_n_iter', 40)
        svr_n_iter = config.get('svr_n_iter', 50)
        mlp_n_iter = config.get('mlp_n_iter', 30)
        xgb_search_space = config.get('xgb_search_space', None)
        rf_search_space = config.get('rf_search_space', None)
        svr_search_space = config.get('svr_search_space', None)
        mlp_search_space = config.get('mlp_search_space', None)
    else:
        xgb_n_iter, rf_n_iter, svr_n_iter, mlp_n_iter = 60, 40, 50, 30
        xgb_search_space = rf_search_space = svr_search_space = mlp_search_space = None
    
    print("=" * 50)
    print("未发现缓存，开始训练模型...")
    print("=" * 50)
    
    print("\n[1/5] XGBoost...")
    xgb_model = optimize_xgboost(X_train, y_train, random_state, 
                                  n_iter=xgb_n_iter, search_space=xgb_search_space)
    
    print("\n[2/5] 随机森林...")
    rf_model = optimize_random_forest(X_train, y_train, random_state, 
                                       n_iter=rf_n_iter, search_space=rf_search_space)
    
    print("\n[3/5] 支持向量机（标准化数据）...")
    svr_model = optimize_svr(X_train_s, y_train, random_state, 
                              n_iter=svr_n_iter, search_space=svr_search_space)
    
    print("\n[4/5] 岭回归...")
    ridge_model = optimize_ridge(X_train, y_train, random_state)
    
    print("\n[5/5] 神经网络（标准化数据）...")
    mlp_model = optimize_mlp(X_train_s, y_train, random_state, 
                              n_iter=mlp_n_iter, search_space=mlp_search_space)
    
    models = {
        'XGBoost': xgb_model,
        '随机森林': rf_model,
        '支持向量机': svr_model,
        '岭回归': ridge_model,
        '神经网络': mlp_model
    }
    
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