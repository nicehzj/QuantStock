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
@njit
def fix_weights_ashare(ideal_w, is_limit_up, is_limit_down, is_suspended, force_liquidate):
    """路径依赖的 A 股权重修正器"""
    actual_w = np.zeros_like(ideal_w)
    for c in range(ideal_w.shape[1]):
        if not (is_suspended[0, c] or force_liquidate[0, c]):
            actual_w[0, c] = ideal_w[0, c]
            
    for t in range(1, ideal_w.shape[0]):
        for c in range(ideal_w.shape[1]):
            target = ideal_w[t, c]
            prev = actual_w[t-1, c]
            if force_liquidate[t, c]:
                actual_w[t, c] = 0.0
            elif is_suspended[t, c]:
                actual_w[t, c] = prev
            elif is_limit_up[t, c] and target > prev:
                actual_w[t, c] = prev
            elif is_limit_down[t, c] and target < prev:
                actual_w[t, c] = prev
            else:
                actual_w[t, c] = target
    return actual_w


class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.03):
        """计算回测核心指标 (带防崩保护)"""
        if returns.empty or returns.std() == 0:
            return {
                'TotalReturn': 0.0, 'AnnReturn': 0.0, 'MaxDD': 0.0,
                'Sharpe': -9.0, 'Calmar': 0.0, 'Vol': 0.0,
                'WinRate': 0.0, 'PLRatio': 0.0
            }
            
        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        running_max = cum_returns.expanding().max()
        max_dd_val = ((cum_returns - running_max) / (running_max + 1e-9)).min()
        vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / (vol + 1e-9)
        calmar = ann_return / (abs(max_dd_val) + 1e-9)
        win_rate = (returns > 0).sum() / len(returns)
        pos_rets = returns[returns > 0]; neg_rets = returns[returns < 0]
        profit_loss_ratio = pos_rets.mean() / (abs(neg_rets.mean()) + 1e-9) if not neg_rets.empty else 0

        return {
            'TotalReturn': total_return, 'AnnReturn': ann_return, 'MaxDD': max_dd_val,
            'Sharpe': sharpe, 'Calmar': calmar, 'Vol': vol,
            'WinRate': win_rate, 'PLRatio': profit_loss_ratio
        }


class AshareVectorizedOptimizer:
    def __init__(self):
        self.engine = FactorEngine()
        
    def _clean_and_prepare_data(self, price_df, volume_df, preclose_df):
        limit_ratios = pd.Series(0.099, index=price_df.columns)
        limit_ratios[price_df.columns.str.startswith('sz.30') | price_df.columns.str.startswith('sh.688')] = 0.199
        is_limit_up = price_df >= (preclose_df * (1 + limit_ratios) - 0.002)
        is_limit_down = price_df <= (preclose_df * (1 - limit_ratios) + 0.002)
        is_suspended = volume_df <= 0
        force_liquidate = price_df.isna() & preclose_df.isna()
        valid_price = price_df.ffill().bfill().clip(lower=0.001).fillna(0.001)
        valid_preclose = preclose_df.ffill().bfill().clip(lower=0.001).fillna(0.001)
        return valid_price, valid_preclose, is_limit_up.values, is_limit_down.values, is_suspended.values, force_liquidate.values

    def run_fast_optimize(self, factor_name, mode='smoke'):
        # 1. 准备参数空间
        if mode == 'smoke':
            windows = [10, 20]; price_fields = ['open']; freqs = ['5D']
        else:
            windows = [5, 10, 20, 40, 60]; price_fields = ['open', 'vwap']; freqs = ['1D', '5D', '10D']

        # 2. 加载基础数据并【统一转换为 DatetimeIndex】(关键修复点)
        big_df_sample = self.engine.calculate_all_variants(factor_name, windows)
        variants = {}
        for w in windows:
            base_col = [c for c in big_df_sample.columns if f"_{w}" in c][0]
            pivot = self.engine.get_pivoted_factor(base_col, factor_name=factor_name, windows=windows, fill_na=False)
            pivot.index = pd.to_datetime(pivot.index)
            m = pivot.median(axis=1); mad = (pivot.sub(m, axis=0)).abs().median(axis=1)
            variants[w] = pivot.clip(m - 4.4478*mad, m + 4.4478*mad, axis=0).astype(np.float32)
            
        prices_dict = {}
        for pf in price_fields:
            df_p = self.engine.get_pivoted_factor(pf, fill_na=True)
            df_p.index = pd.to_datetime(df_p.index)
            prices_dict[pf] = df_p
            
        volume = self.engine.get_pivoted_factor('volume', fill_na=False).fillna(0)
        volume.index = pd.to_datetime(volume.index)
        
        pre_close = self.engine.get_pivoted_factor('pre_close', fill_na=True)
        pre_close.index = pd.to_datetime(pre_close.index)

        # 3. 创建目录
        output_dir = Path(f"data/opt_{factor_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> 深度寻优启动！结果归档至: {output_dir}")

        all_params = []
        for w in windows:
            for pf in price_fields:
                for freq in freqs:
                    for d in [True, False]:
                        all_params.append({'w':w, 'p':pf, 'f':freq, 'd':d})

        master_results = []
        for i, p in enumerate(all_params):
            p_name = f"W{p['w']}_P{p['p']}_F{p['f']}_D{p['d']}"
            print(f"[{i+1}/{len(all_params)}] 正在评估组合并生成分析图: {p_name}")
            
            f_pivot = variants[p['w']]
            o_p = prices_dict[p['p']]
            
            # 执行深度评估 (历史滚动 + 诊断图)
            rolling_df, final_metrics = self.run_deep_evaluation(
                factor_name, f_pivot, o_p, volume, pre_close, p, p_name, output_dir
            )
            
            master_results.append({**p, **final_metrics})
            gc.collect()

        # 5. 保存排行榜
        master_df = pd.DataFrame(master_results).sort_values('Sharpe', ascending=False)
        master_df.to_csv(output_dir / "master_leaderboard.csv", index=False)
        
        best = master_df.iloc[0]
        print(f"\n[OK] 寻优完成。冠军组合: W:{best['w']} P:{best['p']} F:{best['f']} D:{best['d']} Sharpe:{best['Sharpe']:.4f}")
        
        # 6. 执行冠军 OOS 盲测
        self.run_final_oos_best(factor_name, variants[best['w']], prices_dict[best['p']], volume, pre_close, best, output_dir)

    def run_deep_evaluation(self, factor_name, f_pivot, o_price, volume, pre_close, p, p_name, output_dir):
        # 1. 全样本严谨回测
        _, final_metrics = self.rigorous_backtest_single(f_pivot, o_price, volume, pre_close, freq=p['f'], ascending=p['d'])
        
        # 2. 历史滚动测试
        windows_dates = []
        for y in range(2006, 2023): windows_dates.append((f"{y}-01-01", f"{y+2}-12-31"))
            
        rolling_results = []
        for start, end in windows_dates:
            try:
                slice_f = f_pivot.loc[start:end]; slice_p = o_price.loc[start:end]
                slice_v = volume.loc[start:end]; slice_pre = pre_close.loc[start:end]
                if len(slice_f) < 100: continue
                
                res, m = self.rigorous_backtest_single(slice_f, slice_p, slice_v, slice_pre, freq=p['f'], ascending=p['d'])
                
                years = len(slice_f) / 252; records = res.orders.records_readable
                trade_vol = (records['Size'].abs() * records['Price']).sum() if not records.empty else 0
                avg_val = res.value().mean()
                turnover = (trade_vol / 2) / (avg_val + 1e-9) / years
                rolling_results.append({'Window': f"{start[:4]}-{end[:4]}", 'Turnover': turnover, **m})
            except Exception: continue
            
        rolling_df = pd.DataFrame(rolling_results)
        if not rolling_df.empty:
            rolling_df.to_csv(output_dir / f"rolling_{p_name}.csv", index=False)
            # 绘图逻辑
            fig, axes = plt.subplots(2, 2, figsize=(16, 10)); plt.subplots_adjust(hspace=0.35)
            axes[0,0].bar(rolling_df['Window'], rolling_df['Sharpe'], color='skyblue', label='Sharpe')
            axes[0,0].plot(rolling_df['Window'], rolling_df['Calmar'], marker='o', color='gold', label='Calmar')
            axes[0,0].set_title(f'稳定性诊断: {p_name}'); axes[0,0].legend(); axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,1].plot(rolling_df['Window'], rolling_df['WinRate'], marker='o', color='forestgreen'); axes[0,1].set_title('胜率走势 (%)'); axes[0,1].tick_params(axis='x', rotation=45)
            axes[1,0].bar(rolling_df['Window'], rolling_df['MaxDD'] * 100, color='salmon'); axes[1,0].set_title('各窗口最大回撤 (%)'); axes[1,0].invert_yaxis(); axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,1].bar(rolling_df['Window'], rolling_df['Turnover'], color='gray', alpha=0.6); axes[1,1].set_title('年化单边换手率'); axes[1,1].tick_params(axis='x', rotation=45)
            plt.savefig(output_dir / f"diagnostic_{p_name}.png"); plt.close(fig)
            
        return rolling_df, final_metrics

    def run_final_oos_best(self, factor_name, f_pivot, o_price, volume, pre_close, best_p, output_dir):
        """冠军 OOS 战报绘图"""
        test_start = '2024-01-01'
        print(f">>> 正在对冠军参数进行 OOS 终极盲测 (2024-至今)...")
        slice_f = f_pivot.loc[test_start:]; slice_p = o_price.loc[test_start:]
        slice_v = volume.loc[test_start:]; slice_pre = pre_close.loc[test_start:]
        
        portfolio, metrics = self.rigorous_backtest_single(slice_f, slice_p, slice_v, slice_pre, freq=best_p['f'], ascending=best_p['d'])
        
        # 绘图
        plt.figure(figsize=(12, 7))
        strategy_equity = portfolio.value()
        norm_equity = strategy_equity / strategy_equity.iloc[0]
        norm_equity.plot(label='Strategy (Best)', color='red', lw=2)
        
        benchmark_p = self.engine.get_benchmark_prices(start_date=test_start)
        if not benchmark_p.empty:
            common_idx = norm_equity.index.intersection(benchmark_p.index)
            bench_norm = benchmark_p.loc[common_idx] / benchmark_p.loc[common_idx].iloc[0]
            bench_norm.plot(label='Benchmark (HS300)', color='black', alpha=0.5, linestyle='--')
            
        plt.title(f'Final OOS Test (2024-Present): {factor_name}\nBest: W{best_p["w"]} P:{best_p["p"]} F:{best_p["f"]}')
        plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(output_dir / "final_oos_best_equity.png"); plt.close()
        print(f"[OK] 冠军 OOS 战报已生成至: {output_dir}")

    def rigorous_backtest_single(self, f_pivot, o_p, volume, pre_close, top_n=20, freq='5D', ascending=True, slippage=0.0005):
        """核心回测引擎: 已统一日期索引类型"""
        # 1. 严格对齐日期索引
        common_idx = f_pivot.index.intersection(o_p.index).intersection(volume.index).intersection(pre_close.index)
        if len(common_idx) < 10:
            return None, PerformanceAnalyzer.empty_metrics()
            
        f_sub = f_pivot.loc[common_idx]; o_sub = o_p.loc[common_idx]
        v_sub = volume.loc[common_idx]; pr_sub = pre_close.loc[common_idx]
        
        # 2. 准备约束
        valid_p, valid_pre, l_up, l_down, susp, force_liq = self._clean_and_prepare_data(o_sub, v_sub, pr_sub)
        
        # 3. 信号与排名 (活跃度过滤)
        active_factor = f_sub.where((v_sub > 0) & (f_sub != 0), np.nan)
        ideal_w_raw = (active_factor.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
        
        step = int(freq.replace('D','')); rebalance_days = ideal_w_raw.index[::step]
        ideal_w = ideal_w_raw.reindex(ideal_w_raw.index).loc[rebalance_days].reindex(ideal_w_raw.index).ffill().shift(1).fillna(0.0)
        
        # 4. 执行状态机
        actual_w_vals = fix_weights_ashare(ideal_w.values, l_up, l_down, susp, force_liq)
        actual_w = pd.DataFrame(actual_w_vals, index=ideal_w.index, columns=ideal_w.columns)
        
        # 5. VectorBT 回测
        base_rate = 0.0003
        portfolio = vbt.Portfolio.from_orders(valid_p, size=actual_w, size_type='targetpercent', init_cash=1000000, fees=base_rate, slippage=slippage, freq='1D', cash_sharing=True)
        
        # 6. 精准成本扣除 (5元门槛费 + 印花税)
        records = portfolio.orders.records_readable
        if not records.empty:
            date_col = 'Timestamp' if 'Timestamp' in records.columns else ('Index' if 'Index' in records.columns else 'Date')
            sell_mask = records['Size'] < 0
            records['StampDuty'] = records.loc[sell_mask, 'Size'].abs() * records.loc[sell_mask, 'Price'] * 0.001
            records['MinSurcharge'] = (5.0 - records['Size'].abs() * records['Price'] * base_rate).clip(lower=0)
            
            costs = records.groupby(pd.to_datetime(records[date_col]).dt.date)[['StampDuty', 'MinSurcharge']].sum().sum(axis=1)
            costs.index = pd.to_datetime(costs.index)
            cumulative_cost = costs.reindex(portfolio.value().index).fillna(0).cumsum()
            true_returns = (portfolio.value() - cumulative_cost).pct_change().fillna(0)
        else:
            true_returns = portfolio.returns().fillna(0)
            
        return portfolio, PerformanceAnalyzer.calculate_metrics(true_returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 因子深度全景优化器')
    parser.add_argument('--factor', type=str, default='smart_money_proxy', help='因子名称')
    parser.add_argument('--mode', type=str, choices=['smoke', 'full'], default='smoke', help='运行模式: smoke(冒烟) 或 full(全量)')
    args = parser.parse_args()
    t_start = time.time()
    AshareVectorizedOptimizer().run_fast_optimize(args.factor, mode=args.mode)
    print(f"\n任务完成！总耗时: {time.time() - t_start:.2f} 秒\n")