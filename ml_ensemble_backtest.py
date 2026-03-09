import pandas as pd
import numpy as np
import vectorbt as vbt
import os
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
from sklearn.ensemble import RandomForestRegressor
import time

class MLAdvancedBacktester:
    """
    机器学习多因子集成回测系统
    【核心理念】：
    单一因子可能失效，但通过机器学习（随机森林、XGBoost等）融合多个低相关因子，
    可以构建出更加稳健的“超级因子”。
    """
    def __init__(self):
        self.engine = FactorEngine()
        # 精选 10 个具有不同预测逻辑的因子作为特征 (Features)
        self.selected_factors = [
            'smart_money_proxy', 'amihud_illiq', 'skew_20', 'ivol_20',
            'alpha12', 'alpha46', 'rsv_20', 'buy_vol_ratio', 
            'vol_shock_alpha', 'price_efficiency'
        ]
        
    def prepare_data(self):
        """
        数据准备：计算特征因子和预测目标 (Label)。
        """
        print(f">>> 1. 正在计算原始因子 (2006-2026)...")
        # 调用引擎获取多因子宽表
        df = self.engine.calculate_factors(factor_list=self.selected_factors, start_date='2006-01-01')
        df['date'] = pd.to_datetime(df['date'])
        
        print(">>> 2. 计算预测目标 (Target = 未来 5 日收益率)...")
        # 机器学习的目标通常是预测未来一段周期的涨跌幅
        df = df.sort_values(['code', 'date'])
        df['target'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        
        # 移除空值（主要是最后 5 天没有 target）
        df = df.dropna(subset=self.selected_factors + ['target'])
        return df

    def _safe_rank_transform(self, df):
        """
        截面标准化：确保模型只根据当天的排名优劣来学习。
        防止“均值回归”类的特征在不同时期因绝对值大小不同而干扰模型。
        """
        temp_df = df.copy()
        for factor in self.selected_factors:
            # groupby('date') 是关键：只在每天内部做百分比排名
            temp_df[factor] = temp_df.groupby('date')[factor].rank(pct=True)
        return temp_df

    def run_segmented_ml(self, data):
        """
        分段训练逻辑：引入 Purging (净化) 和 Embargo (隔离带)。
        【量化 AI 的大坑】：
        如果训练集和测试集靠得太近，由于收益率计算是重叠的，会导致模型“偷看答案”。
        本函数通过强制时间留白，确保回测结果的真实性。
        """
        print(">>> 3. 开始执行分段训练 (安全去污模式)...")
        
        # 定义历史上的不同经济周期进行训练和测试
        segments = [
            {'name': 'Cycle_1', 'train': ('2006-01-01', '2012-12-31'), 'test': ('2013-06-01', '2015-12-31')},
            {'name': 'Cycle_2', 'train': ('2016-01-01', '2022-12-31'), 'test': ('2023-06-01', '2026-12-31')}
        ]
        
        all_oos_results = []
        
        for seg in segments:
            print(f"\n--- 阶段: {seg['name']} ---")
            
            # --- 训练集预处理 ---
            train_raw = data[(data['date'] >= seg['train'][0]) & (data['date'] <= seg['train'][1])].copy()
            
            # 1. Purging (净化)：移除训练集最后 20 天，防止特征泄露
            last_train_date = train_raw['date'].max()
            train_raw = train_raw[train_raw['date'] < (last_train_date - pd.Timedelta(days=22))]
            
            # 2. IID 采样：每 5 天取一个样本，消除金融时间序列的高度自相关性
            train_raw['rank_in_code'] = train_raw.groupby('code').cumcount()
            train_raw = train_raw[train_raw['rank_in_code'] % 5 == 0]
            
            # 3. 截面标准化
            train_df = self._safe_rank_transform(train_raw)
            
            # --- 测试集预处理 ---
            test_raw = data[(data['date'] >= seg['test'][0]) & (data['date'] <= seg['test'][1])].copy()
            test_df = self._safe_rank_transform(test_raw)
            
            print(f"有效训练样本: {len(train_df)} | 测试样本: {len(test_df)}")
            
            # 4. 训练随机森林回归模型
            # 限制树的深度和叶子节点最小样本数，严防“过拟合”
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=200, 
                n_jobs=-1, 
                random_state=42
            )
            model.fit(train_df[self.selected_factors], train_df['target'])
            
            # 5. 产生预测信号 (集成因子)
            test_df = test_df.copy()
            test_df['ensemble_factor'] = model.predict(test_df[self.selected_factors])
            all_oos_results.append(test_df)
            
        return pd.concat(all_oos_results)

    def run_backtest(self, oos_data, buy_top=20, hold_top=40, stop_loss=0.15):
        """
        样本外 (Out-of-Sample) 资金曲线模拟。
        包含周一调仓、强制止损、仓位平权等实盘细节。
        """
        print(f"\n>>> 4. 启动【去污版】样本外回测...")
        
        # 转换为宽表矩阵格式
        price_df = oos_data.pivot(index='date', columns='code', values='close').ffill()
        factor_pivot = oos_data.pivot(index='date', columns='code', values='ensemble_factor')
        ranks = factor_pivot.rank(axis=1, ascending=False, method='first')
        
        target_weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
        current_holdings = {} # 记录当前持仓
        rebalance_days = ranks.index[ranks.index.weekday == 0] # 每周一调仓
        
        for i, date in enumerate(ranks.index):
            today_prices = price_df.loc[date]
            today_ranks = ranks.loc[date]
            
            # 1. 每日止损逻辑：亏损超过 15% 立即出场
            to_stop_loss = [code for code, ep in current_holdings.items() if today_prices[code]/ep - 1 <= -stop_loss]
            for code in to_stop_loss: del current_holdings[code]
            
            # 2. 定期调仓逻辑
            if date in rebalance_days:
                # 剔除排名跌出前 40 的旧票
                still_holding = {code: ep for code, ep in current_holdings.items() if today_ranks[code] <= hold_top}
                # 补位：买入排名进入前 20 的新票
                if len(still_holding) < buy_top:
                    candidates = today_ranks[today_ranks <= buy_top].sort_values().index
                    for cand in candidates:
                        if len(still_holding) >= buy_top: break
                        if cand not in still_holding:
                            still_holding[cand] = today_prices[cand]
                current_holdings = still_holding
            
            # 设置权重 (分配 95% 资金，防滑点)
            if current_holdings:
                w = 0.95 / len(current_holdings)
                for code in current_holdings: target_weights.at[date, code] = w

        # 执行回测
        portfolio = vbt.Portfolio.from_orders(
            price_df, size=target_weights.shift(1).fillna(0.0), size_type='targetpercent',
            init_cash=1000000, fees=0.0013, cash_sharing=True, group_by=True, freq='1D'
        )
        
        print(portfolio.stats())
        # 绘图保存结果
        plt.figure(figsize=(12, 6))
        portfolio.value().plot(label='ML Ensemble Strategy', color='purple', lw=2)
        plt.title('ML Integration Backtest Result')
        plt.savefig('data/ml_strictly_purged_report.png')
        return portfolio

if __name__ == "__main__":
    tester = MLAdvancedBacktester()
    all_data = tester.prepare_data()
    oos_predictions = tester.run_segmented_ml(all_data)
    tester.run_backtest(oos_predictions)
