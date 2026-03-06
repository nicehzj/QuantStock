import vectorbt as vbt
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
import argparse

class MultiBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        
    def run_comparison(self, factors=None, top_n=20, start_date='2021-01-01', freq='1D'):
        if factors is None:
            factors = ['smart_money_proxy', 'amihud_illiq', 'skew_20']

        all_factors_df = self.engine.calculate_factors(factor_list=factors, start_date=start_date)
        all_factors_df['date'] = pd.to_datetime(all_factors_df['date'])
        
        sql = f"SELECT date, code, open, close, volume, preclose FROM read_parquet('{self.engine.parquet_glob}') WHERE date >= '{start_date}'"
        raw_df = self.engine.db.execute(sql).df()
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        # 同时提取 Open 和 Close 用于敏感性测试
        self.close_df = raw_df.pivot(index='date', columns='code', values='close').ffill()
        self.open_df = raw_df.pivot(index='date', columns='code', values='open').ffill()
        self.volume_df = raw_df.pivot(index='date', columns='code', values='volume').fillna(0.0)
        self.preclose_df = raw_df.pivot(index='date', columns='code', values='preclose').ffill()
        
        benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        self.benchmark_returns = benchmark_df.set_index('date')['close'].pct_change().fillna(0.0)

        self.performance_summary = []

        for factor_name in factors:
            factor_data = all_factors_df.pivot(index='date', columns='code', values=factor_name)
            self._backtest_one_logic(factor_data, factor_name, start_date, top_n, ascending=False, freq=freq)
            # 自动反转逻辑
            self._backtest_one_logic(factor_data, f"{factor_name}(Rev)", start_date, top_n, ascending=True, freq=freq)

        perf_df = pd.DataFrame(self.performance_summary).sort_values('Sharpe', ascending=False)
        print("\n" + "="*110)
        print(f">>> 终极严谨版因子绩效排行榜 (频率: {freq} | 成交基准: T+1 Open)")
        print("="*110)
        pd.options.display.float_format = '{:.4f}'.format
        print(perf_df.to_string(index=False))
        return perf_df

    def _backtest_one_logic(self, factor_pivot, display_name, start_date, top_n, ascending, freq):
        try:
            common_dates = self.close_df.index.intersection(factor_pivot.index)
            common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
            
            c_price = self.close_df.loc[common_dates]
            o_price = self.open_df.loc[common_dates]
            c_volume = self.volume_df.loc[common_dates]
            c_preclose = self.preclose_df.loc[common_dates]
            c_factor = factor_pivot.reindex(index=common_dates, columns=c_price.columns)
            
            # --- 1. A 股实盘约束逻辑 (修正版) ---
            limit_ratios = pd.Series(0.099, index=c_price.columns)
            limit_ratios[c_price.columns.str.startswith('sz.30') | c_price.columns.str.startswith('sh.688')] = 0.199
            is_limit_up = o_price >= (c_preclose * (1 + limit_ratios)).round(3)
            is_limit_down = o_price <= (c_preclose * (1 - limit_ratios)).round(3)
            is_suspended = c_volume <= 0
            
            # 2. 计算理想排名信号 (T 日收盘选股)
            is_in_top_n = c_factor.rank(axis=1, ascending=ascending, method='first') <= top_n
            ideal_target_w = is_in_top_n.shift(1).fillna(False).astype(float) * (0.95 / top_n)
            
            # 3. 采样频率处理
            step = int(freq.replace('D','')) if 'D' in freq else 1
            sample_mask = np.zeros(len(ideal_target_w), dtype=bool)
            sample_mask[::step] = True
            
            # 4. 【核心重构】迭代式权重修正 (处理卖不出、买不进的真实困境)
            actual_weights = np.zeros_like(ideal_target_w.values)
            target_vals = ideal_target_w.values
            l_up = is_limit_up.values
            l_down = is_limit_down.values
            susp = is_suspended.values
            
            # 初始状态
            for t in range(1, len(ideal_target_w)):
                prev_w = actual_weights[t-1]
                # 只有在采样日才更新目标，否则维持
                target_w = target_vals[t] if sample_mask[t] else prev_w
                
                # 逐股检查约束
                row_w = np.copy(prev_w)
                for col in range(len(prev_w)):
                    curr_target = target_w[col]
                    curr_prev = prev_w[col]
                    
                    if curr_target > curr_prev: # 增仓/买入
                        if not (l_up[t, col] or susp[t, col]):
                            row_w[col] = curr_target
                    elif curr_target < curr_prev: # 减仓/卖出
                        if not (l_down[t, col] or susp[t, col]):
                            row_w[col] = curr_target
                actual_weights[t] = row_w
            
            final_actual_weights = pd.DataFrame(actual_weights, index=common_dates, columns=c_price.columns)
            
            # 5. 执行 VectorBT (使用 T+1 开盘价成交，模拟非对称费率)
            # 为模拟非对称费率，我们取买卖平均并略微调高以保持保守性
            portfolio = vbt.Portfolio.from_orders(
                o_price, size=final_actual_weights, size_type='targetpercent',
                init_cash=1000000, fees=0.0015, # 调高至 0.15% 覆盖非对称印花税的保守估算
                cash_sharing=True, group_by=True, freq='1D'
            )
            
            # 计算指标
            strat_returns = portfolio.returns().fillna(0.0)
            mkt_returns = self.benchmark_returns.reindex(strat_returns.index).fillna(0.0)
            beta = np.cov(strat_returns, mkt_returns)[0, 1] / (np.var(mkt_returns) + 1e-9)
            alpha = (strat_returns.mean() - beta * mkt_returns.mean()) * 252
            
            stats = portfolio.stats()
            self.performance_summary.append({
                'Factor': display_name,
                'Return[%]': stats['Total Return [%]'],
                'Sharpe': stats['Sharpe Ratio'],
                'MaxDD[%]': stats['Max Drawdown [%]'],
                'Alpha(Ann)': alpha,
                'Beta': beta,
                'Trades': stats['Total Trades']
            })
        except Exception as e:
            print(f"[Error] {display_name} 失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', nargs='+', default=None)
    parser.add_argument('--freq', type=str, default='5D')
    args = parser.parse_args()
    MultiBacktester().run_comparison(factors=args.factors, freq=args.freq)
