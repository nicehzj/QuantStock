import pandas as pd
import numpy as np
import vectorbt as vbt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from factor_engine import FactorEngine
from tqdm import tqdm
import gc
import argparse
import time
from numba import njit
from pathlib import Path

# 设置绘图风格 (解决中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# ✨ 核心引擎: Numba 加速的 A 股实盘交易约束
# =====================================================================
# 为什么要用 @njit？
# A 股有涨跌停限制（10% 或 20%），这会导致“想买买不到”或“想卖卖不出”。
# 这种“路径依赖”的仿真逻辑在纯 Python 循环中极其缓慢。
# @njit 会将此函数直接编译成机器码，使回测速度提升数百倍。
@njit
def fix_weights_ashare(ideal_w, is_limit_up, is_limit_down, is_suspended, force_liquidate):
    """
    根据 A 股实际规则修正理想权重。
    ideal_w: 算法算出的理想仓位 (例如：平权买入前 20 名)
    is_limit_up/down: 涨跌停掩码
    """
    actual_w = np.zeros_like(ideal_w)
    # 初始化第一天仓位
    for c in range(ideal_w.shape[1]):
        if not (is_suspended[0, c] or force_liquidate[0, c]):
            actual_w[0, c] = ideal_w[0, c]
            
    # 逐日推演真实仓位
    for t in range(1, ideal_w.shape[0]):
        for c in range(ideal_w.shape[1]):
            target = ideal_w[t, c] # 目标仓位
            prev = actual_w[t-1, c] # 昨日真实持仓
            
            if force_liquidate[t, c]:
                actual_w[t, c] = 0.0 # 强制平仓 (如退市)
            elif is_suspended[t, c]:
                actual_w[t, c] = prev # 停牌了，想卖卖不掉，想买买不进，维持持仓
            elif is_limit_up[t, c] and target > prev:
                actual_w[t, c] = prev # 涨停了，想加仓加不进，维持原持仓
            elif is_limit_down[t, c] and target < prev:
                actual_w[t, c] = prev # 跌停了，想减仓减不掉，维持原持仓
            else:
                actual_w[t, c] = target # 正常交易
    return actual_w


class PerformanceAnalyzer:
    """绩效评估工具类：计算年化收益、Sharpe、最大回撤等核心指标"""
    @staticmethod
    def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.03):
        if returns.empty or returns.std() == 0:
            return {'Sharpe': -9.0, 'MaxDD': 0.0}
            
        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算最大回撤 (Max Drawdown)
        running_max = cum_returns.expanding().max()
        max_dd_val = ((cum_returns - running_max) / (running_max + 1e-9)).min()
        
        vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / (vol + 1e-9)
        calmar = ann_return / (abs(max_dd_val) + 1e-9)
        
        return {
            'TotalReturn': total_return, 
            'AnnReturn': ann_return, 
            'MaxDD': max_dd_val,
            'Sharpe': sharpe, 
            'Calmar': calmar, 
            'Vol': vol,
            'WinRate': (returns > 0).sum() / len(returns)
        }


class AshareVectorizedOptimizer:
    """
    A 股向量化参数优化器
    自动化探索：哪个时间窗口(W)、哪个价格(P)、哪个调仓频率(F)才是因子的“黄金组合”。
    """
    def __init__(self):
        self.engine = FactorEngine()
        
    def _clean_and_prepare_data(self, price_df, volume_df, preclose_df):
        """生成交易限制掩码"""
        limit_ratios = pd.Series(0.099, index=price_df.columns)
        # 创业板、科创板限制为 20%
        limit_ratios[price_df.columns.str.startswith('sz.30') | price_df.columns.str.startswith('sh.688')] = 0.199
        
        # 判定涨跌停 (考虑一定的小数误差)
        is_limit_up = price_df >= (preclose_df * (1 + limit_ratios) - 0.002)
        is_limit_down = price_df <= (preclose_df * (1 - limit_ratios) + 0.002)
        is_suspended = volume_df <= 0
        force_liquidate = price_df.isna() & preclose_df.isna()
        
        # 填充价格用于计算收益
        valid_price = price_df.ffill().bfill().clip(lower=0.001).fillna(0.001)
        valid_preclose = preclose_df.ffill().bfill().clip(lower=0.001).fillna(0.001)
        return valid_price, valid_preclose, is_limit_up.values, is_limit_down.values, is_suspended.values, force_liquidate.values

    def run_fast_optimize(self, factor_name, mode='smoke'):
        """
        启动网格寻优。
        mode='smoke': 只跑少量参数，快速检查逻辑是否跑通。
        mode='full': 全量搜索最优参数。
        """
        # 1. 设置搜索空间
        if mode == 'smoke':
            windows = [10, 20]; price_fields = ['open']; freqs = ['5D']
        else:
            windows = [5, 10, 20, 40, 60]; price_fields = ['open', 'vwap']; freqs = ['1D', '5D', '10D']

        # 2. 预先计算因子变体矩阵
        big_df_sample = self.engine.calculate_all_variants(factor_name, windows)
        variants = {}
        for w in windows:
            base_col = [c for c in big_df_sample.columns if f"_{w}" in c][0]
            pivot = self.engine.get_pivoted_factor(base_col, factor_name=factor_name, windows=windows, fill_na=False)
            pivot.index = pd.to_datetime(pivot.index)
            # 对因子进行 MAD 去极值，防止个别离群值破坏截面排名
            m = pivot.median(axis=1); mad = (pivot.sub(m, axis=0)).abs().median(axis=1)
            variants[w] = pivot.clip(m - 4.4478*mad, m + 4.4478*mad, axis=0).astype(np.float32)
            
        prices_dict = {pf: self.engine.get_pivoted_factor(pf, fill_na=True) for pf in price_fields}
        for df in prices_dict.values(): df.index = pd.to_datetime(df.index)
            
        volume = self.engine.get_pivoted_factor('volume', fill_na=False).fillna(0)
        volume.index = pd.to_datetime(volume.index)
        
        pre_close = self.engine.get_pivoted_factor('pre_close', fill_na=True)
        pre_close.index = pd.to_datetime(pre_close.index)

        output_dir = Path(f"data/opt_{factor_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成参数排列组合
        all_params = []
        for w in windows:
            for pf in price_fields:
                for freq in freqs:
                    for d in [True, False]:
                        all_params.append({'w':w, 'p':pf, 'f':freq, 'd':d})

        master_results = []
        for i, p in enumerate(all_params):
            p_name = f"W{p['w']}_P{p['p']}_F{p['f']}_D{p['d']}"
            print(f"[{i+1}/{len(all_params)}] 正在深度评估: {p_name}")
            
            # 运行包含滚动窗口 (Rolling) 诊断的评估流程
            rolling_df, final_metrics = self.run_deep_evaluation(
                factor_name, variants[p['w']], prices_dict[p['p']], volume, pre_close, p, p_name, output_dir
            )
            master_results.append({**p, **final_metrics})
            gc.collect()

        # 保存结果排行榜
        master_df = pd.DataFrame(master_results).sort_values('Sharpe', ascending=False)
        master_df.to_csv(output_dir / "master_leaderboard.csv", index=False)
        
        print(f"\n[OK] 寻优完成。冠军参数组合的 Sharpe: {master_df.iloc[0]['Sharpe']:.4f}")

    def run_deep_evaluation(self, factor_name, f_pivot, o_price, volume, pre_close, p, p_name, output_dir):
        """
        深度评估模块：除了全样本回测，还进行滚动回测以验证因子的时间稳定性。
        如果一个因子只在 2015 年牛市厉害，其他时间拉胯，那就是过拟合。
        """
        # 全样本回测
        _, final_metrics = self.rigorous_backtest_single(f_pivot, o_price, volume, pre_close, freq=p['f'], ascending=p['d'])
        
        # 分年份滚动测试
        rolling_results = []
        for y in range(2006, 2023):
            start, end = f"{y}-01-01", f"{y+2}-12-31" # 测试 3 年窗口
            try:
                res, m = self.rigorous_backtest_single(f_pivot.loc[start:end], o_price.loc[start:end], volume.loc[start:end], pre_close.loc[start:end], freq=p['f'], ascending=p['d'])
                rolling_results.append({'Window': f"{y}-{y+2}", **m})
            except: continue
            
        # 自动生成性能诊断图并保存 (省略绘图细节)...
        return pd.DataFrame(rolling_results), final_metrics

    def rigorous_backtest_single(self, f_pivot, o_p, volume, pre_close, top_n=20, freq='5D', ascending=True, slippage=0.0005):
        """
        核心回测逻辑：严格对齐、排名、生成信号、修正约束、VectorBT 执行、精准计费。
        这是整个系统逻辑最严密的地方。
        """
        common_idx = f_pivot.index.intersection(o_p.index)
        f_sub = f_pivot.loc[common_idx]; o_sub = o_p.loc[common_idx]
        v_sub = volume.loc[common_idx]; pr_sub = pre_close.loc[common_idx]
        
        # 截面排名：剔除停牌股，在活跃股中选前 N 名
        active_factor = f_sub.where((v_sub > 0) & (f_sub != 0), np.nan)
        ideal_w_raw = (active_factor.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
        
        # 确定调仓日期并处理 T+1 效应（shift(1) 代表今天算出来的信号，明天执行）
        step = int(freq.replace('D','')); rebalance_days = ideal_w_raw.index[::step]
        ideal_w = ideal_w_raw.reindex(ideal_w_raw.index).loc[rebalance_days].reindex(ideal_w_raw.index).ffill().shift(1).fillna(0.0)
        
        # 准备约束掩码
        valid_p, _, l_up, l_down, susp, force_liq = self._clean_and_prepare_data(o_sub, v_sub, pr_sub)
        # Numba 修正真实持仓权重
        actual_w_vals = fix_weights_ashare(ideal_w.values, l_up, l_down, susp, force_liq)
        actual_w = pd.DataFrame(actual_w_vals, index=ideal_w.index, columns=ideal_w.columns)
        
        # VectorBT 核心执行引擎
        portfolio = vbt.Portfolio.from_orders(valid_p, size=actual_w, size_type='targetpercent', init_cash=1000000, fees=0.0003, slippage=slippage, freq='1D', cash_sharing=True)
        
        # 结果计算
        return portfolio, PerformanceAnalyzer.calculate_metrics(portfolio.returns())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 因子参数自动寻优实验室')
    parser.add_argument('--factor', type=str, default='smart_money_proxy', help='因子名称')
    parser.add_argument('--mode', type=str, choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    
    AshareVectorizedOptimizer().run_fast_optimize(args.factor, mode=args.mode)
