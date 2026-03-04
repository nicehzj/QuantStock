import vectorbt as vbt
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
from scipy import stats

class MultiBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        
    def run_comparison(self, factors=None, top_n=20, start_date='2021-01-01'):
        """
        运行多因子对比回测。默认运行全部 20 个因子。
        """
        if factors is None:
            factors = [
                'alpha12', 'alpha6', 'alpha46', 'overnight_ret', 'turn_vol_20', 
                'amihud_illiq', 'ivol_20', 'smart_money_proxy', 'skew_20', 'vol_shock_sue',
                'price_efficiency', 'open_vol_corr', 'rsv_20', 'pv_momentum', 'z_score_20',
                'buy_vol_ratio', 'kurt_20', 'rank_pv_corr', 'ma_dispersion', 'days_to_high'
            ]

        # 1. 计算所有因子 (传入 start_date 过滤内存)
        all_factors_df = self.engine.calculate_factors(factor_list=factors, start_date=start_date)
        all_factors_df['date'] = pd.to_datetime(all_factors_df['date'])
        
        # 2. 获取行情数据 (增加 SQL 过滤)
        sql = f"SELECT date, code, close, volume, preclose FROM read_parquet('{self.engine.parquet_glob}') WHERE date >= '{start_date}'"
        raw_df = self.engine.db.execute(sql).df()
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        self.price_df = raw_df.pivot(index='date', columns='code', values='close').ffill()
        self.volume_df = raw_df.pivot(index='date', columns='code', values='volume').fillna(0.0)
        self.preclose_df = raw_df.pivot(index='date', columns='code', values='preclose').ffill()
        
        benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        self.benchmark_returns = benchmark_df.set_index('date')['close'].pct_change().fillna(0.0)

        self.results_equity = {}
        self.performance_summary = []

        for factor_name in factors:
            factor_data = all_factors_df.pivot(index='date', columns='code', values=factor_name)
            
            # 1. 尝试原始正向 (Largest first)
            print(f">>> 测试正向逻辑 (Descending): {factor_name}")
            stats_orig = self._backtest_one_logic(factor_data, factor_name, start_date, top_n, ascending=False)
            
            # 2. 如果表现太差（收益率为负），自动尝试反向 (Smallest first)
            if stats_orig and stats_orig['Return[%]'] < 0:
                print(f"    [检测到反向潜力] 正在测试反转逻辑: {factor_name}(Rev)")
                self._backtest_one_logic(factor_data, f"{factor_name}(Rev)", start_date, top_n, ascending=True)

        # 汇总与绘图
        perf_df = pd.DataFrame(self.performance_summary).sort_values('Alpha(Ann)', ascending=False)
        print("\n" + "="*100)
        print(">>> 因子绩效大比拼 (包含反转尝试)")
        print("="*100)
        pd.options.display.float_format = '{:.4f}'.format
        print(perf_df.to_string(index=False))
        
        # 绘图：展示 Alpha 最强的前 8 名
        plt.figure(figsize=(16, 8))
        top_factors = perf_df.head(8)['Factor'].tolist()
        for f in top_factors:
            self.results_equity[f].plot(label=f)
        
        # 增加基准
        common_dates = self.price_df.index[self.price_df.index >= pd.to_datetime(start_date)]
        benchmark_equity = (benchmark_df.set_index('date')['close'].reindex(common_dates).ffill().bfill())
        (benchmark_equity / benchmark_equity.iloc[0] * 1000000).plot(label='HS300 Benchmark', color='black', lw=3, ls='--')
        
        plt.title('Top Alpha Factors (Original vs Reversed) vs HS300 (2021-2026)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('data/reversed_factor_comparison.png')
        print(f"\n[OK] 深度比对报告已生成: data/reversed_factor_comparison.png")
        return perf_df

    def _backtest_one_logic(self, factor_pivot, display_name, start_date, top_n, ascending):
        try:
            common_dates = self.price_df.index.intersection(factor_pivot.index)
            common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
            
            curr_price = self.price_df.loc[common_dates]
            curr_volume = self.volume_df.loc[common_dates]
            curr_preclose = self.preclose_df.loc[common_dates]
            curr_factor = factor_pivot.reindex(index=common_dates, columns=curr_price.columns)
            
            is_in_top_n = curr_factor.where(curr_price > 0).rank(axis=1, ascending=ascending, method='first') <= top_n
            target_in_top = is_in_top_n.shift(1).fillna(False)
            
            limit_ratios = pd.Series(0.099, index=curr_price.columns)
            limit_ratios[curr_price.columns.str.startswith('sz.30') | curr_price.columns.str.startswith('sh.688')] = 0.199
            is_limit_up = curr_price >= (curr_preclose * (1 + limit_ratios)).round(3)
            is_limit_down = curr_price <= (curr_preclose * (1 - limit_ratios)).round(3)
            is_suspended = curr_volume <= 0
            
            target_weights = pd.DataFrame(0.0, index=common_dates, columns=curr_price.columns)
            target_weights[target_in_top & (~is_limit_up) & (~is_suspended)] = 0.95 / top_n
            
            portfolio = vbt.Portfolio.from_orders(
                curr_price, size=target_weights, size_type='targetpercent',
                init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True,
                call_seq='auto', freq='1D'
            )
            
            # 计算 Alpha/Beta
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
                'Beta': beta
            }
            self.performance_summary.append(res_stats)
            return res_stats
        except:
            return None

if __name__ == "__main__":
    import sys
    # 如果命令行有参数，则使用命令行指定的因子；否则运行默认全部因子
    factors_to_test = sys.argv[1:] if len(sys.argv) > 1 else None
    
    if factors_to_test:
        print(f">>> 接收到自定义回测任务: {factors_to_test}")
    
    MultiBacktester().run_comparison(factors=factors_to_test)
