import vectorbt as vbt
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from factor_engine import FactorEngine

class VBTBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        
    def run_momentum_screening(self, top_n=20, fees=0.0013, start_date='2021-01-01', end_date=None):
        start_time = time.time()
        factors_df = self.engine.calculate_basic_factors()
        sql = f"SELECT date, code, close, volume, preclose FROM read_parquet('{self.engine.parquet_glob}')"
        raw_df = self.engine.db.execute(sql).df()
        
        # 统一日期格式
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        
        # 构建对齐宽表
        price_df = raw_df.pivot(index='date', columns='code', values='close').ffill().fillna(0.0)
        volume_df = raw_df.pivot(index='date', columns='code', values='volume').fillna(0.0)
        preclose_df = raw_df.pivot(index='date', columns='code', values='preclose').ffill().fillna(0.0)
        
        factor_pivot = factors_df.pivot(index='date', columns='code', values='factor_mom_20')
        
        # 取交集
        common_dates = price_df.index.intersection(factor_pivot.index)
        common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
        if end_date: common_dates = common_dates[common_dates <= pd.to_datetime(end_date)]
        
        price_df = price_df.loc[common_dates]
        volume_df = volume_df.loc[common_dates]
        preclose_df = preclose_df.loc[common_dates]
        factor_pivot = factor_pivot.reindex(index=common_dates, columns=price_df.columns)
        
        # 信号
        is_in_top_n = factor_pivot.where(price_df > 0).rank(axis=1, ascending=True, method='first') <= top_n
        target_in_top = is_in_top_n.shift(1).fillna(False)
        
        limit_ratios = pd.Series(0.099, index=price_df.columns)
        limit_ratios[price_df.columns.str.startswith('sz.30') | price_df.columns.str.startswith('sh.688')] = 0.199
        is_limit_up = price_df >= (preclose_df * (1 + limit_ratios)).round(3)
        is_limit_down = price_df <= (preclose_df * (1 - limit_ratios)).round(3)
        is_suspended = volume_df <= 0
        
        # 构建 target_weights (from_orders 版)
        target_weights = pd.DataFrame(index=common_dates, columns=price_df.columns, dtype=float)
        
        # 买入信号：只要在 Top-N 且未涨停且未停牌
        buy_mask = target_in_top & (~is_limit_up) & (~is_suspended)
        target_weights[buy_mask] = 0.95 / top_n
        
        # 卖出信号：只要不在 Top-N 且未跌停 (不管停牌，强行释放资金以维持回测继续)
        sell_mask = (~target_in_top) & (~is_limit_down)
        target_weights[sell_mask] = 0.0
        
        print(f">>> 启动终极回测 (日期区间: {common_dates.min()} 至 {common_dates.max()})")
        portfolio = vbt.Portfolio.from_orders(
            price_df, 
            size=target_weights, 
            size_type='targetpercent',
            init_cash=1000000, fees=fees, cash_sharing=True, group_by=True,
            call_seq='auto'
        )
        
        equity = portfolio.value()
        print(f"最终账户价值: {equity.iloc[-1]:,.2f}")
        
        # --- 增加基准对比 ---
        plt.figure(figsize=(12, 7))
        plt.style.use('ggplot')
        
        # 绘制策略曲线
        equity.plot(label='Strategy (Momentum Top-20)', color='forestgreen', lw=2)
        
        # 加载并绘制沪深 300 基准
        try:
            benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df = benchmark_df.set_index('date')['close'].reindex(common_dates).ffill().bfill()
            # 归一化：将指数起点对齐到 1,000,000 初始资金
            benchmark_equity = benchmark_df / benchmark_df.iloc[0] * 1000000
            benchmark_equity.plot(label='Benchmark (HS300)', color='gray', lw=1.5, ls='--')
        except Exception as e:
            print(f"[Warning] 基准数据加载失败: {e}")
            
        plt.title('Strategy vs Benchmark: Equity Curve (2021-2026)')
        plt.xlabel('Date')
        plt.ylabel('Total Value (CNY)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/final_corrected_report.png')
        print("Report saved: data/final_corrected_report.png")
        return portfolio

if __name__ == "__main__":
    VBTBacktester().run_momentum_screening()
