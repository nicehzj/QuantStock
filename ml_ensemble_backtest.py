import pandas as pd
import numpy as np
import vectorbt as vbt
import os
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
from sklearn.ensemble import RandomForestRegressor
import time

class MLAdvancedBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        self.selected_factors = [
            'smart_money_proxy', 'amihud_illiq', 'skew_20', 'ivol_20',
            'alpha12', 'alpha46', 'rsv_20', 'buy_vol_ratio', 
            'vol_shock_sue', 'price_efficiency'
        ]
        
    def prepare_data(self):
        print(f">>> 1. 正在计算精选 10 因子 (2006-2026)...")
        df = self.engine.calculate_factors(factor_list=self.selected_factors, start_date='2006-01-01')
        df['date'] = pd.to_datetime(df['date'])
        
        print(">>> 2. 执行横截面 Rank 标准化...")
        df = df.sort_values(['code', 'date'])
        df['target'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        df = df.dropna(subset=self.selected_factors + ['target'])
        
        for factor in self.selected_factors:
            df[factor] = df.groupby('date')[factor].rank(pct=True)
            
        return df

    def run_segmented_ml(self, data):
        print(">>> 3. 开始执行分段训练 (Purging Gap 安全模式)...")
        segments = [
            {'name': 'Cycle_1', 'train': ('2006-01-01', '2012-12-31'), 'test': ('2013-06-01', '2015-12-31')},
            {'name': 'Cycle_2', 'train': ('2016-01-01', '2022-12-31'), 'test': ('2023-06-01', '2026-12-31')}
        ]
        all_oos_results = []
        for seg in segments:
            print(f"\n--- 阶段: {seg['name']} ---")
            train_df = data[(data['date'] >= seg['train'][0]) & (data['date'] <= seg['train'][1])].copy()
            last_train_date = train_df['date'].max()
            train_df = train_df[train_df['date'] < (last_train_date - pd.Timedelta(days=10))]
            
            test_df = data[(data['date'] >= seg['test'][0]) & (data['date'] <= seg['test'][1])].copy()
            
            model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=100, n_jobs=-1, random_state=42)
            model.fit(train_df[self.selected_factors], train_df['target'])
            
            test_df['ensemble_factor'] = model.predict(test_df[self.selected_factors])
            all_oos_results.append(test_df)
            
        return pd.concat(all_oos_results)

    def run_backtest(self, oos_data, buy_top=20, hold_top=40, stop_loss=0.15):
        print(f"\n>>> 4. 启动【缓冲带+单股止损】回测 (SL: {stop_loss*100}%)...")
        
        price_df = oos_data.pivot(index='date', columns='code', values='close').ffill()
        factor_pivot = oos_data.pivot(index='date', columns='code', values='ensemble_factor')
        ranks = factor_pivot.rank(axis=1, ascending=False, method='first')
        
        target_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
        current_holdings = {} 
        rebalance_days = ranks.index[ranks.index.weekday == 0]
        
        print(">>> 正在模拟复杂实盘逻辑 (缓冲带 + 每日止损监控)...")
        for i, date in enumerate(ranks.index):
            today_prices = price_df.loc[date]
            today_ranks = ranks.loc[date]
            
            # A. 每日止损
            to_stop_loss = [code for code, ep in current_holdings.items() if today_prices[code]/ep - 1 <= -stop_loss]
            for code in to_stop_loss: del current_holdings[code]
            
            # B. 周一调仓
            if date in rebalance_days:
                still_holding = {code: ep for code, ep in current_holdings.items() if today_ranks[code] <= hold_top}
                if len(still_holding) < buy_top:
                    candidates = today_ranks[today_ranks <= buy_top].sort_values().index
                    for cand in candidates:
                        if len(still_holding) >= buy_top: break
                        if cand not in still_holding:
                            still_holding[cand] = today_prices[cand]
                current_holdings = still_holding
            
            # C. 设置权重
            if current_holdings:
                w = 0.95 / len(current_holdings)
                for code in current_holdings: target_weights.at[date, code] = w

        portfolio = vbt.Portfolio.from_orders(
            price_df, size=target_weights.shift(1).fillna(0.0), size_type='targetpercent',
            init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True, freq='1D'
        )
        
        print("\n" + "="*60)
        print(">>> 终极封测报告 (审计通过版)")
        print("="*60)
        print(portfolio.stats())
        
        plt.figure(figsize=(12, 6))
        portfolio.value().plot(label='Safe ML Strategy', color='firebrick')
        
        try:
            benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df = benchmark_df.set_index('date')['close'].reindex(portfolio.wrapper.index).ffill().bfill()
            benchmark_normalized = (benchmark_df / benchmark_df.iloc[0] * 1000000)
            benchmark_normalized.plot(label='HS300 Benchmark', color='black', ls='--')
        except: pass
        
        plt.title('Final ML Strategy: Purged OOS + Buffer-Rank + StopLoss')
        plt.savefig('data/ml_final_sealed_report.png')
        return portfolio

if __name__ == "__main__":
    tester = MLAdvancedBacktester()
    all_data = tester.prepare_data()
    oos_predictions = tester.run_segmented_ml(all_data)
    tester.run_backtest(oos_predictions)
