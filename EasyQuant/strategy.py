# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd

class StrategyBase(ABC):
    """策略基类"""
    def __init__(self, data_loader=None):
        self.data_loader = data_loader

    @abstractmethod
    def get_rebalance_dates(self, start_date: str, end_date: str) -> List[str]:
        """确定调仓日期（如每周、每月）"""
        pass

    @abstractmethod
    def select_stocks(self, date: str) -> List[str]:
        """选股逻辑"""
        pass

    @abstractmethod
    def get_target_weights(self, date: str, selected_stocks: List[str]) -> Dict[str, float]:
        """获取目标权重 (如: {'sh.600000': 0.1, 'sz.000001': 0.1})"""
        pass

class MonthlyRebalanceStrategy(StrategyBase):
    """内置的每月初调仓基类"""
    def get_rebalance_dates(self, start_date, end_date):
        all_dates = self.data_loader.get_trading_dates(start_date, end_date)
        monthly_first = []
        last_month = None
        for d in all_dates:
            curr_month = d[:7] # YYYY-MM
            if curr_month != last_month:
                monthly_first.append(d)
                last_month = curr_month
        return monthly_first
