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
        # 精选 10 个低相关性因子
        self.selected_factors = [
            'smart_money_proxy', 'amihud_illiq', 'skew_20', 'ivol_20',
            'alpha12', 'alpha46', 'rsv_20', 'buy_vol_ratio', 
            'vol_shock_alpha', 'price_efficiency'
        ]
        
    def prepare_data(self):
        print(f">>> 1. 正在计算原始因子 (2006-2026)...")
        # 此时只拿原始值，不进行任何全局标准化
        df = self.engine.calculate_factors(factor_list=self.selected_factors, start_date='2006-01-01')
        df['date'] = pd.to_datetime(df['date'])
        
        print(">>> 2. 计算预测目标 (Target = Forward 5D Return)...")
        df = df.sort_values(['code', 'date'])
        df['target'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        
        # 移除包含 NaN 的行 (主要是最后 5 天没有 target)
        df = df.dropna(subset=self.selected_factors + ['target'])
        return df

    def _safe_rank_transform(self, df):
        """
        严格截面标准化：确保只在当天的截面上独立计算排名百分比
        """
        temp_df = df.copy()
        for factor in self.selected_factors:
            # 这里的 groupby('date') 确保了没有跨日的信息泄露
            temp_df[factor] = temp_df.groupby('date')[factor].rank(pct=True)
        return temp_df

    def run_segmented_ml(self, data):
        """
        实施分段训练逻辑：加入 Purging, Embargo 和 IID 采样
        """
        print(">>> 3. 开始执行分段训练 (安全净化模式)...")
        
        # 定义阶段，留出足够的 Buffer (Embargo Gap)
        # Gap 必须大于 Max(Lookback(20), Lookforward(5))
        segments = [
            {'name': 'Cycle_1', 'train': ('2006-01-01', '2012-12-31'), 'test': ('2013-06-01', '2015-12-31')},
            {'name': 'Cycle_2', 'train': ('2016-01-01', '2022-12-31'), 'test': ('2023-06-01', '2026-12-31')}
        ]
        
        all_oos_results = []
        
        for seg in segments:
            print(f"\n--- 阶段: {seg['name']} ---")
            
            # --- 训练集预处理 ---
            train_raw = data[(data['date'] >= seg['train'][0]) & (data['date'] <= seg['train'][1])].copy()
            
            # 1. Purging: 移除训练集末尾 20 天，防止与测试集特征重叠
            last_train_date = train_raw['date'].max()
            train_raw = train_raw[train_raw['date'] < (last_train_date - pd.Timedelta(days=22))]
            
            # 2. IID 采样: 每 5 天取一个样本，消除标签重叠导致的自相关
            # 选取每个代码的每隔 5 条记录
            train_raw['rank_in_code'] = train_raw.groupby('code').cumcount()
            train_raw = train_raw[train_raw['rank_in_code'] % 5 == 0]
            
            # 3. 实时 Rank 标准化 (只在当前训练集内部按天 Rank)
            train_df = self._safe_rank_transform(train_raw)
            
            # --- 测试集预处理 ---
            test_raw = data[(data['date'] >= seg['test'][0]) & (data['date'] <= seg['test'][1])].copy()
            test_df = self._safe_rank_transform(test_raw) # 同样只在测试集内部按天独立 Rank
            
            print(f"有效训练样本(IID): {len(train_df)} | 测试样本: {len(test_df)}")
            
            # 模型训练 (强剪枝限制过拟合)
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=200, 
                n_jobs=-1, 
                random_state=42
            )
            model.fit(train_df[self.selected_factors], train_df['target'])
            
            # 预测信号
            test_df = test_df.copy()
            test_df['ensemble_factor'] = model.predict(test_df[self.selected_factors])
            all_oos_results.append(test_df)
            
        return pd.concat(all_oos_results)

    def run_backtest(self, oos_data, buy_top=20, hold_top=40, stop_loss=0.15):
        print(f"\n>>> 4. 启动【去污版】样本外回测...")
        
        price_df = oos_data.pivot(index='date', columns='code', values='close').ffill()
        factor_pivot = oos_data.pivot(index='date', columns='code', values='ensemble_factor')
        ranks = factor_pivot.rank(axis=1, ascending=False, method='first')
        
        target_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
        current_holdings = {} 
        rebalance_days = ranks.index[ranks.index.weekday == 0]
        
        for i, date in enumerate(ranks.index):
            today_prices = price_df.loc[date]
            today_ranks = ranks.loc[date]
            
            # 1. 每日止损
            to_stop_loss = [code for code, ep in current_holdings.items() if today_prices[code]/ep - 1 <= -stop_loss]
            for code in to_stop_loss: del current_holdings[code]
            
            # 2. 周一调仓
            if date in rebalance_days:
                still_holding = {code: ep for code, ep in current_holdings.items() if today_ranks[code] <= hold_top}
                if len(still_holding) < buy_top:
                    candidates = today_ranks[today_ranks <= buy_top].sort_values().index
                    for cand in candidates:
                        if len(still_holding) >= buy_top: break
                        if cand not in still_holding:
                            still_holding[cand] = today_prices[cand]
                current_holdings = still_holding
            
            # 3. 设置权重
            if current_holdings:
                w = 0.95 / len(current_holdings)
                for code in current_holdings: target_weights.at[date, code] = w

        portfolio = vbt.Portfolio.from_orders(
            price_df, size=target_weights.shift(1).fillna(0.0), size_type='targetpercent',
            init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True, freq='1D'
        )
        
        print("\n" + "="*60)
        print(">>> 终极封测报告 (完全去污版)")
        print("="*60)
        print(portfolio.stats())
        
        plt.figure(figsize=(12, 6))
        portfolio.value().plot(label='Decoupled ML Strategy', color='purple', lw=2)
        plt.title('Strictly Decoupled & Purged OOS Performance')
        plt.savefig('data/ml_strictly_purged_report.png')
        return portfolio

if __name__ == "__main__":
    tester = MLAdvancedBacktester()
    all_data = tester.prepare_data()
    oos_predictions = tester.run_segmented_ml(all_data)
    tester.run_backtest(oos_predictions)
