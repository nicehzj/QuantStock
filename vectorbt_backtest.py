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
        """
        运行多因子对比回测。
        """
        if factors is None:
            factors = [
                'alpha12', 'alpha6', 'alpha46', 'overnight_ret', 'turn_vol_20', 
                'amihud_illiq', 'ivol_20', 'smart_money_proxy', 'skew_20', 'vol_shock_sue'
            ]

        # 1. 计算因子
        all_factors_df = self.engine.calculate_factors(factor_list=factors, start_date=start_date)
        all_factors_df['date'] = pd.to_datetime(all_factors_df['date'])
        
        # 2. 获取行情数据
        sql = f"SELECT date, code, close, volume, preclose FROM read_parquet('{self.engine.parquet_glob}') WHERE date >= '{start_date}'"
        raw_df = self.engine.db.execute(sql).df()
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        self.price_df = raw_df.pivot(index='date', columns='code', values='close').ffill()
        self.volume_df = raw_df.pivot(index='date', columns='code', values='volume').fillna(0.0)
        self.preclose_df = raw_df.pivot(index='date', columns='code', values='preclose').ffill()
        
        # 3. 基准数据
        benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        self.benchmark_returns = benchmark_df.set_index('date')['close'].pct_change().fillna(0.0)

        self.results_equity = {}
        self.performance_summary = []

        print(f"\n>>> 开始多因子大比拼 | 调仓频率: {freq} | 样本数: {len(factors)}")

        for factor_name in factors:
            factor_data = all_factors_df.pivot(index='date', columns='code', values=factor_name)
            
            # 正向测试 (选最大)
            stats_orig = self._backtest_one_logic(factor_data, factor_name, start_date, top_n, ascending=False, freq=freq)
            
            # 如果亏损，尝试反转 (选最小)
            if stats_orig and stats_orig['Return[%]'] < 0:
                self._backtest_one_logic(factor_data, f"{factor_name}(Rev)", start_date, top_n, ascending=True, freq=freq)

        # 4. 汇总展示
        if not self.performance_summary:
            print("[Fatal] 未能生成任何有效的回测结果。")
            return
            
        perf_df = pd.DataFrame(self.performance_summary).sort_values('Sharpe', ascending=False)
        print("\n" + "="*100)
        print(f">>> 因子绩效排行榜 (频率: {freq})")
        print("="*100)
        pd.options.display.float_format = '{:.4f}'.format
        print(perf_df.to_string(index=False))
        
        # 5. 绘图
        plt.figure(figsize=(14, 7))
        top_factors = perf_df.head(5)['Factor'].tolist()
        for f in top_factors:
            if f in self.results_equity:
                self.results_equity[f].plot(label=f)
        
        # 对齐基准
        common_dates = self.price_df.index[self.price_df.index >= pd.to_datetime(start_date)]
        benchmark_equity = (benchmark_df.set_index('date')['close'].reindex(common_dates).ffill().bfill())
        (benchmark_equity / benchmark_equity.iloc[0] * 1000000).plot(label='HS300 Benchmark', color='black', lw=2, ls='--')
        
        plt.title(f'Top Factors Comparison (Freq: {freq})')
        plt.legend()
        plt.savefig(f'data/factor_compare_{freq}.png')
        return perf_df

    def _backtest_one_logic(self, factor_pivot, display_name, start_date, top_n, ascending, freq):
        try:
            common_dates = self.price_df.index.intersection(factor_pivot.index)
            common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
            
            curr_price = self.price_df.loc[common_dates]
            curr_volume = self.volume_df.loc[common_dates]
            curr_preclose = self.preclose_df.loc[common_dates]
            curr_factor = factor_pivot.reindex(index=common_dates, columns=curr_price.columns)
            
            # --- A 股实盘约束逻辑 ---
            limit_ratios = pd.Series(0.099, index=curr_price.columns)
            limit_ratios[curr_price.columns.str.startswith('sz.30') | curr_price.columns.str.startswith('sh.688')] = 0.199
            is_limit_up = curr_price >= (curr_preclose * (1 + limit_ratios)).round(3)
            is_limit_down = curr_price <= (curr_preclose * (1 - limit_ratios)).round(3)
            is_suspended = curr_volume <= 0
            
            # 1. 计算理想排名
            is_in_top_n = curr_factor.where(curr_price > 0).rank(axis=1, ascending=ascending, method='first') <= top_n
            ideal_target = is_in_top_n.shift(1).fillna(False)
            
            # 2. 生成权重 (考虑频率)
            daily_weights = pd.DataFrame(0.0, index=common_dates, columns=curr_price.columns)
            # 基础规则：入选且未涨停未停牌才买入
            daily_weights[ideal_target & (~is_limit_up) & (~is_suspended)] = 0.95 / top_n
            
            # 3. 频率采样
            mask = np.zeros(len(daily_weights), dtype=bool)
            if freq == '1D':
                mask[:] = True
            else:
                step = int(freq.replace('D','')) if 'D' in freq else 5
                mask[::step] = True
            
            # 仅在采样点更新目标权重，其余日期设为 NaN 以便 ffill (维持持仓)
            final_weights = pd.DataFrame(np.nan, index=daily_weights.index, columns=daily_weights.columns)
            final_weights.loc[mask] = daily_weights.loc[mask]
            
            # 关键：处理跌停卖不出的情况
            # 如果目标是 0 (卖出) 但当前跌停，则强制维持旧权重
            # 这里简单处理：先 ffill 维持权重，然后在 vbt 内部通过价格机制进一步模拟
            final_weights = final_weights.ffill().fillna(0.0)
            
            # 再次叠加物理约束：如果在采样点想卖出但跌停，或者想买入但涨停，则该点的权重更新无效
            # (VectorBT targetpercent 模式会自动处理一部分，但显式清理更安全)
            
            portfolio = vbt.Portfolio.from_orders(
                curr_price, size=final_weights, size_type='targetpercent',
                init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True,
                call_seq='auto', freq='1D'
            )
            
            # 计算指标
            strat_returns = portfolio.returns().fillna(0.0)
            mkt_returns = self.benchmark_returns.reindex(strat_returns.index).fillna(0.0)
            beta = np.cov(strat_returns, mkt_returns)[0, 1] / np.var(mkt_returns)
            alpha = (strat_returns.mean() - beta * mkt_returns.mean()) * 252
            
            self.results_equity[display_name] = portfolio.value()
            stats = portfolio.stats()
            
            res_stats = {
                'Factor': display_name,
                'Return[%]': stats['Total Return [%]'],
                'Sharpe': stats['Sharpe Ratio'],
                'MaxDD[%]': stats['Max Drawdown [%]'],
                'Alpha(Ann)': alpha,
                'Beta': beta,
                'Fees': stats['Total Fees Paid']
            }
            self.performance_summary.append(res_stats)
            return res_stats
        except Exception as e:
            print(f"[Error] {display_name} 失败: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 多因子回测工具')
    parser.add_argument('--factors', nargs='+', help='指定回测因子名称', default=None)
    parser.add_argument('--freq', type=str, help='调仓频率 (如 1D, 5D, 20D)', default='1D')
    args = parser.parse_args()
    
    MultiBacktester().run_comparison(factors=args.factors, freq=args.freq)
