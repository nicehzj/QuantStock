# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from .data_loader import LocalDataLoader
from .strategy import StrategyBase

@dataclass
class TradeRecord:
    date: str
    code: str
    direction: str
    volume: int
    price: float
    cost: float

class BacktestEngine:
    def __init__(self, 
                 initial_cash: float = 1000000, 
                 commission: float = 0.0003, # 佣金万三
                 slippage: float = 0.001,    # 滑点千一
                 data_loader: LocalDataLoader = None):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.data_loader = data_loader
        
        # 状态
        self.positions = {} # {code: volume}
        self.trade_history = []
        self.equity_history = [] # [(date, total_value)]

    def run(self, strategy: StrategyBase, start_date: str, end_date: str):
        strategy.data_loader = self.data_loader
        rebalance_dates = strategy.get_rebalance_dates(start_date, end_date)
        all_trading_days = self.data_loader.get_trading_dates(start_date, end_date)
        
        print(f"[Engine] 开始回测 {start_date} ~ {end_date}")
        
        for date in all_trading_days:
            # 1. 调仓逻辑
            if date in rebalance_dates:
                selected_stocks = strategy.select_stocks(date)
                target_weights = strategy.get_target_weights(date, selected_stocks)
                self._rebalance(date, target_weights)
            
            # 2. 每日市值统计
            market_value = self._get_market_value(date)
            total_value = self.cash + market_value
            self.equity_history.append({"date": date, "total_value": total_value, "cash": self.cash})

    def _get_market_value(self, date: str) -> float:
        total = 0.0
        for code, volume in self.positions.items():
            price = self.data_loader.get_nearest_price(code, date)
            if price:
                total += price * volume
        return total

    def _rebalance(self, date: str, target_weights: Dict[str, float]):
        """核心调仓：将持仓推向目标权重"""
        current_total_value = self.cash + self._get_market_value(date)
        
        # A. 卖出不再目标列表的
        for code in list(self.positions.keys()):
            if code not in target_weights:
                self._execute_trade(date, code, -self.positions[code])
        
        # B. 调整权重 (先卖后买，释放现金)
        orders = []
        for code, weight in target_weights.items():
            target_value = current_total_value * weight
            price = self.data_loader.get_nearest_price(code, date)
            if not price: continue
            
            curr_volume = self.positions.get(code, 0)
            target_volume = int(target_value / price / 100) * 100
            diff = target_volume - curr_volume
            if diff != 0:
                orders.append((code, diff))
        
        # 按卖出在前排序以防可用现金不足
        orders.sort(key=lambda x: x[1]) 
        for code, diff in orders:
            self._execute_trade(date, code, diff)

    def _execute_trade(self, date: str, code: str, volume: int):
        if volume == 0: return
        
        price = self.data_loader.get_nearest_price(code, date)
        if not price: return
        
        # 计算执行价 (含滑点)
        exec_price = price * (1 + self.slippage) if volume > 0 else price * (1 - self.slippage)
        amount = exec_price * volume
        fee = abs(amount * self.commission)
        total_cost = amount + fee # 买入为正，卖出为负(加负数即扣钱)
        
        if self.cash < total_cost and volume > 0:
            # 简单降级：现金不足，不买
            return 

        self.cash -= total_cost
        self.positions[code] = self.positions.get(code, 0) + volume
        if self.positions[code] <= 0:
            del self.positions[code]
            
        self.trade_history.append(TradeRecord(date, code, "BUY" if volume > 0 else "SELL", abs(volume), exec_price, fee))

    def get_results(self) -> pd.DataFrame:
        df = pd.DataFrame(self.equity_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
