import pandas as pd
import numpy as np
import vectorbt as vbt
import os
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
from sklearn.ensemble import RandomForestRegressor
import time

class MLEnsembleBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        # 仅使用前 3 个因子打通流程
        self.top_factors = ['smart_money_proxy', 'amihud_illiq', 'skew_20']
        
    def prepare_data(self):
        print(f">>> 1. 正在准备长周期数据 (2006-2026) [{self.top_factors}]...")
        # 显式传入 2006-01-01 作为起始日期
        df = self.engine.calculate_factors(factor_list=self.top_factors, start_date='2006-01-01')
        df['date'] = pd.to_datetime(df['date'])
        
        # 计算目标：未来 5 日收益 (Forward Return)
        print(">>> 2. 计算训练目标 (Forward 5D Returns)...")
        df = df.sort_values(['code', 'date'])
        df['target'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        
        # 清洗：移除任何包含 NaN 的行
        df = df.dropna(subset=self.top_factors + ['target'])
        
        # 横截面标准化 (Rank)
        print(">>> 3. 执行横截面 Rank 标准化...")
        for factor in self.top_factors:
            df[factor] = df.groupby('date')[factor].rank(pct=True)
            
        return df

    def train_and_predict(self, data):
        print(">>> 4. 机器学习模型训练 (样本区间: 2006-2020)...")
        # 严格的时间切分
        train_mask = (data['date'] >= '2006-01-01') & (data['date'] <= '2020-12-31')
        train_df = data[train_mask]
        
        if train_df.empty:
            raise ValueError("训练集为空，请检查数据是否包含 2021 年之前的数据。")

        X_train = train_df[self.top_factors]
        y_train = train_df['target']
        
        # 使用随机森林进行非线性合成
        model = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        print(">>> 5. 生成预测信号...")
        # 对全量数据生成预测（实际上我们主要关注 2021 年以后的）
        data['ensemble_factor'] = model.predict(data[self.top_factors])
        
        # 打印特征重要性
        importances = pd.Series(model.feature_importances_, index=self.top_factors)
        print("\n--- 因子重要性 (Feature Importance) ---")
        print(importances)
        
        return data

    def run_backtest(self, data, top_n=20, start_date='2021-01-01'):
        print(f"\n>>> 6. 启动样本外回测 (区间: {start_date} - 2026)...")
        
        # 仅取 2021 年以后的数据进行回测验证
        backtest_data = data[data['date'] >= start_date].copy()
        
        # 构建对齐宽表
        price_df = backtest_data.pivot(index='date', columns='code', values='close').ffill()
        factor_pivot = backtest_data.pivot(index='date', columns='code', values='ensemble_factor')
        
        # 信号：买入预测值最高的 top_n 只
        is_in_top_n = factor_pivot.rank(axis=1, ascending=False, method='first') <= top_n
        # T 日信号，T+1 日动作
        target_weights = is_in_top_n.shift(1).fillna(False).astype(float) * (0.95 / top_n)
        
        # 每周一调仓，降低手续费
        rebalance_days = target_weights.index[target_weights.index.weekday == 0]
        final_weights = pd.DataFrame(0.0, index=target_weights.index, columns=target_weights.columns)
        final_weights.loc[rebalance_days] = target_weights.loc[rebalance_days]
        final_weights = final_weights.replace(0.0, np.nan).ffill().fillna(0.0)

        # 运行 VectorBT
        portfolio = vbt.Portfolio.from_orders(
            price_df, size=final_weights, size_type='targetpercent',
            init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True, freq='1D'
        )
        
        # 导出交易记录
        code_map = {i: code for i, code in enumerate(price_df.columns)}
        orders_raw = portfolio.orders.records
        orders_readable = portfolio.orders.records_readable
        unified_log = pd.DataFrame({
            'Timestamp': orders_readable['Timestamp'],
            'Code': orders_raw['col'].map(code_map),
            'Action': orders_readable['Side'],
            'Price': orders_readable['Price'],
            'Size': orders_readable['Size']
        })
        unified_log.to_csv('data/ml_3factor_trade_log.csv', index=False)
        
        # 汇总报告
        stats = portfolio.stats()
        print("\n" + "="*60)
        print(">>> 3 因子 ML 集成绩效报告 (2021-2026 样本外)")
        print("="*60)
        print(stats)
        
        # 绘图
        plt.figure(figsize=(12, 6))
        portfolio.value().plot(label='ML Ensemble (3 Factors - Weekly Rebalance)', color='crimson')
        
        try:
            benchmark_df = pd.read_parquet('data/index_k/sh.000300.parquet')
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df = benchmark_df.set_index('date')['close'].reindex(portfolio.wrapper.index).ffill().bfill()
            (benchmark_df / benchmark_df.iloc[0] * 1000000).plot(label='HS300 Benchmark', color='black', ls='--')
        except: pass
        
        plt.title('ML Ensemble Factor (Trained 06-20, Tested 21-26)')
        plt.legend()
        plt.savefig('data/ml_3factor_report.png')
        print(f"\n[OK] 报告与流水已保存。交易记录: data/ml_3factor_trade_log.csv")
        return portfolio

if __name__ == "__main__":
    tester = MLEnsembleBacktester()
    df = tester.prepare_data()
    df = tester.train_and_predict(df)
    tester.run_backtest(df)
