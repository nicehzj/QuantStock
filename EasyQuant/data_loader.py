# -*- coding: utf-8 -*-
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

class LocalDataLoader:
    """
    本地数据加载器 - 专为 QuantStock 项目设计
    直接读取 data/daily_k/ 目录下的 Parquet 文件
    """
    def __init__(self, db_path: str = ":memory:", data_dir: str = "data/daily_k"):
        self.data_dir = data_dir
        self.con = duckdb.connect(db_path)
        self._init_view()
        
        # 缓存
        self._trading_dates = None

    def _init_view(self):
        """利用 DuckDB 直接挂载所有 Parquet 文件为虚拟表"""
        parquet_path = os.path.join(self.data_dir, "*.parquet")
        # 假设 Parquet 结构包含: date, code, open, high, low, close, volume, amount, adjustflag等
        # 注意：Baostock 导出的列名通常为 date, code, open...
        try:
            self.con.execute(f"CREATE OR REPLACE VIEW daily_k AS SELECT * FROM read_parquet('{parquet_path}')")
            print(f"[DataLoader] 成功挂载数据源: {parquet_path}")
        except Exception as e:
            print(f"[DataLoader] 挂载失败: {e}。请确保 {self.data_dir} 中有数据。")

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历（基于数据中存在的日期）"""
        query = f"""
            SELECT DISTINCT date 
            FROM daily_k 
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
        """
        df = self.con.execute(query).df()
        # 数据已经是 YYYY-MM-DD 字符串，直接返回即可
        return df['date'].tolist()

    def get_prices(self, date: str, codes: List[str] = None) -> pd.DataFrame:
        """获取指定日期全市场或特定股票的价格"""
        where_clause = f"WHERE date = '{date}'"
        if codes:
            codes_str = "','".join(codes)
            where_clause += f" AND code IN ('{codes_str}')"
            
        query = f"SELECT code, close FROM daily_k {where_clause}"
        return self.con.execute(query).df().set_index('code')

    def get_nearest_price(self, code: str, date: str, lookback: int = 10) -> Optional[float]:
        """处理停牌：向前查找最近的一个收盘价"""
        query = f"""
            SELECT close FROM daily_k 
            WHERE code = '{code}' AND date <= '{date}'
            ORDER BY date DESC LIMIT 1
        """
        res = self.con.execute(query).fetchone()
        return res[0] if res else None

    def get_fundamentals(self, date: str, codes: List[str] = None) -> pd.DataFrame:
        """
        获取基本面数据（如市值）
        注意：如果数据里没存市值，这里需要关联您项目中的其他表
        """
        # 这里预留接口，您可以根据实际 Parquet 列名调整
        query = f"SELECT code, close * volume as turnover FROM daily_k WHERE date = '{date}'"
        return self.con.execute(query).df().set_index('code')
