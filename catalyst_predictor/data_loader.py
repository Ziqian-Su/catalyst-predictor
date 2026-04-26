"""
数据加载与预处理模块
功能：加载CSV数据、特征名清理、数据集划分
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_feature_name(name):
    """
    清理特征名，替换特殊字符
    解决XGBoost无法解析 [ ] < > ( ) { } 等字符的问题
    """
    name = str(name)
    for char in ['[', ']', '<', '>', '(', ')', '{', '}']:
        name = name.replace(char, '_')
    if name[0].isdigit():
        name = f'f_{name}'
    return name


def load_data(data_path, target_col='targets', id_col='COFID'):
    """
    加载数据集
    
    参数：
        data_path: CSV文件路径
        target_col: 目标变量列名
        id_col: ID列名（会被删除）
    
    返回：
        X: 特征DataFrame（列名已清理）
        y: 目标变量Series
        features: 原始特征名列表
        name_mapping: 清理名到原始名的映射
    """
    df = pd.read_csv(data_path)
    
    # 提取特征和目标
    drop_cols = [col for col in [id_col, target_col] if col in df.columns]
    X_raw = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # 清理特征名
    original_names = X_raw.columns.tolist()
    cleaned_names = [clean_feature_name(name) for name in original_names]
    name_mapping = dict(zip(cleaned_names, original_names))
    
    # 应用清理后的列名
    X = X_raw.copy()
    X.columns = cleaned_names
    
    print(f"数据加载完成：样本数 {len(df)}，特征数 {len(original_names)}")
    return X, y, original_names, name_mapping


def split_data(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    
    返回：
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"训练集: {X_train.shape[0]} 样本，测试集: {X_test.shape[0]} 样本")
    return X_train, X_test, y_train, y_test