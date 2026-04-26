"""
工具函数模块
提供路径创建、文件保存、绘图样式设置等通用功能
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    """创建目录（如不存在）"""
    os.makedirs(path, exist_ok=True)


def save_csv(df, path, index=False):
    """保存DataFrame为CSV文件（utf-8-sig编码，兼容Excel打开中文）"""
    df.to_csv(path, index=index, encoding='utf-8-sig')


def set_plot_style():
    """设置matplotlib全局绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'


def save_fig(fig, path, dpi=300):
    """保存图片到指定路径"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)