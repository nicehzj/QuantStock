# -*- coding: utf-8 -*-
import sys
import os

# 将根目录加入路径以方便导入 EasyQuant
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EasyQuant.data_loader import LocalDataLoader
from EasyQuant.strategy import MonthlyRebalanceStrategy
from EasyQuant.engine import BacktestEngine

class Top5Strategy(MonthlyRebalanceStrategy):
    """一个简单的策略示例：每月持仓代码排名前5的股票（等权重）"""
    def select_stocks(self, date: str):
        # 实际开发中，这里可以接入 factor_engine.py 计算的因子
        # 这里简单示例：获取当天所有有价格的股票，取前5个
        df_prices = self.data_loader.get_prices(date)
        if df_prices.empty:
            return []
        return df_prices.index.tolist()[:5]

    def get_target_weights(self, date: str, selected_stocks: list):
        if not selected_stocks: return {}
        weight = 1.0 / len(selected_stocks)
        return {code: weight for code in selected_stocks}

def main():
    # 1. 初始化本地数据源 (指向您通过 baostock 下载的数据目录)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/daily_k")
    loader = LocalDataLoader(data_dir=data_dir)
    
    # 2. 初始化引擎
    engine = BacktestEngine(initial_cash=1000000, data_loader=loader)
    
    # 3. 运行回测
    strategy = Top5Strategy()
    engine.run(strategy, start_date="2023-01-01", end_date="2024-01-01")
    
    # 4. 展示结果
    results = engine.get_results()
    if not results.empty:
        print("\n" + "="*50)
        print("回测完成！")
        print(f"最终资产: {results.iloc[-1]['total_value']:,.2f}")
        print(f"累计收益率: {(results.iloc[-1]['total_value']/1000000 - 1):.2%}")
        print("="*50)
        
        # 打印最后几条净值
        print(results.tail())
    else:
        print("回测期间未获得任何数据，请检查数据路径。")

if __name__ == "__main__":
    main()
