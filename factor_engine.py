import pandas as pd
import duckdb
import os
import time
import numpy as np

class FactorEngine:
    def __init__(self, mode='parquet'):
        self.mode = mode
        self.db = duckdb.connect()
        self.data_dir = 'data/daily_k'
        self.index_dir = 'data/index_k'
        self.parquet_glob = os.path.join(self.data_dir, '*.parquet')
        self.index_path = os.path.join(self.index_dir, 'sh.000300.parquet')
        
    def calculate_factors(self, factor_list=None, start_date='2021-01-01'):
        """
        计算复杂因子及其代理指标。
        :param start_date: 数据计算的起始日期，默认为 2021-01-01
        """
        if factor_list is None:
            # 默认计算全部 20 个因子
            factor_list = [
                'alpha12', 'alpha6', 'alpha46', 'overnight_ret', 'turn_vol_20', 
                'amihud_illiq', 'ivol_20', 'smart_money_proxy', 'skew_20', 'vol_shock_sue',
                'price_efficiency', 'open_vol_corr', 'rsv_20', 'pv_momentum', 'z_score_20',
                'buy_vol_ratio', 'kurt_20', 'rank_pv_corr', 'ma_dispersion', 'days_to_high'
            ]
            
        print(f">>> 因子库启动 [计算因子数: {len(factor_list)}] [起始日期: {start_date}]...")
        start_time = time.time()
        
        # ... (中间 SQL 逻辑保持不变) ...
        # 注意：此处由于使用了窗口函数，内部计算还是会遍历全量 Parquet 保证指标连续性
        # 但最终输出的 DataFrame 会根据 start_date 进行过滤

        # 执行合并查询
        final_factor_cols = ", ".join(factor_sqls)
        main_sql = f"SELECT date, code, close, {final_factor_cols} FROM base_calc WHERE date >= '{start_date}'"
        
        df = self.db.execute(main_sql).df()
        
        cost = time.time() - start_time
        print(f"[OK] 20 个复杂因子计算完成。耗时: {cost:.2f}s, 结果集大小: {len(df)} 行。")
        return df

if __name__ == "__main__":
    engine = FactorEngine()
    test_df = engine.calculate_factors()
    print(test_df.tail())
