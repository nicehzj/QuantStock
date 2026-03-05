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
        全量计算 20 大复杂因子及其代理指标
        :param start_date: 数据输出的起始日期
        """
        if factor_list is None:
            factor_list = [
                'alpha12', 'alpha6', 'alpha46', 'overnight_ret', 'turn_vol_20', 
                'amihud_illiq', 'ivol_20', 'smart_money_proxy', 'skew_20', 'vol_shock_sue',
                'price_efficiency', 'open_vol_corr', 'rsv_20', 'pv_momentum', 'z_score_20',
                'buy_vol_ratio', 'kurt_20', 'rank_pv_corr', 'ma_dispersion', 'days_to_high'
            ]
            
        print(f">>> 因子库启动 [计算因子数: {len(factor_list)}] [起始日期: {start_date}]...")
        start_time = time.time()
        
        # 1. 加载行情数据与指数数据
        self.db.execute(f"CREATE OR REPLACE VIEW raw_data AS SELECT * FROM read_parquet('{self.parquet_glob}')")
        self.db.execute(f"CREATE OR REPLACE VIEW idx_data AS SELECT date, close as idx_close FROM read_parquet('{self.index_path}')")

        # 2. 基础预计算视图
        pre_calc_sql = """
        CREATE OR REPLACE VIEW base_calc AS 
        SELECT 
            r.date, r.code, r.open, r.close, r.high, r.low, r.volume, r.amount, r.preclose,
            (r.close / NULLIF(r.preclose, 0) - 1) as ret,
            (idx.idx_close / NULLIF(LAG(idx.idx_close) OVER (ORDER BY idx.date), 0) - 1) as idx_ret,
            r.close - r.preclose as delta_p,
            r.volume - LAG(r.volume) OVER (PARTITION BY r.code ORDER BY r.date) as delta_v,
            (r.open / NULLIF(r.preclose, 0) - 1) as over_ret,
            (ABS(r.close/NULLIF(r.preclose, 0) - 1) / NULLIF(r.amount, 0)) * 1e9 as illiq_val,
            AVG(r.close) OVER (PARTITION BY r.code ORDER BY r.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as ma5,
            AVG(r.close) OVER (PARTITION BY r.code ORDER BY r.date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ma10,
            AVG(r.close) OVER (PARTITION BY r.code ORDER BY r.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20
        FROM raw_data r
        LEFT JOIN idx_data idx ON r.date = idx.date
        """
        self.db.execute(pre_calc_sql)

        # 3. 动态构建 SQL 因子列
        factor_sqls = []
        
        if 'alpha12' in factor_list:
            factor_sqls.append("-1 * CASE WHEN delta_v > 0 THEN 1 WHEN delta_v < 0 THEN -1 ELSE 0 END * delta_p as alpha12")
        if 'alpha6' in factor_list:
            factor_sqls.append("-1 * COVAR_POP(open, volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) / (NULLIF(STDDEV_POP(open) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) * STDDEV_POP(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 0)) as alpha6")
        if 'alpha46' in factor_list:
            factor_sqls.append("-1 * delta_p / NULLIF(high - low, 0) as alpha46")
        if 'overnight_ret' in factor_list:
            factor_sqls.append("AVG(over_ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as overnight_ret")
        if 'turn_vol_20' in factor_list:
            factor_sqls.append("STDDEV_POP(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / NULLIF(AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as turn_vol_20")
        if 'amihud_illiq' in factor_list:
            factor_sqls.append("AVG(illiq_val) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as amihud_illiq")
        if 'ivol_20' in factor_list:
            factor_sqls.append("STDDEV_POP(ret - idx_ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ivol_20")
        if 'smart_money_proxy' in factor_list:
            factor_sqls.append("(amount / NULLIF(volume, 0)) / NULLIF(close, 0) as smart_money_proxy")
        if 'skew_20' in factor_list:
            factor_sqls.append("(AVG(ret*ret*ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) - 3*AVG(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)*AVG(ret*ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) + 2*POWER(AVG(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 3)) / NULLIF(POWER(STDDEV_POP(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 3), 0) as skew_20")
        if 'vol_shock_sue' in factor_list:
            factor_sqls.append("(over_ret * (volume / NULLIF(AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0))) as vol_shock_sue")
        if 'price_efficiency' in factor_list:
            factor_sqls.append("(close - open) / NULLIF(high - low, 0) as price_efficiency")
        if 'open_vol_corr' in factor_list:
            factor_sqls.append("-1 * COVAR_POP(open, volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) / NULLIF(STDDEV_POP(open) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) * STDDEV_POP(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 0) as open_vol_corr")
        if 'rsv_20' in factor_list:
            factor_sqls.append("(close - MIN(low) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)) / NULLIF(MAX(high) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) - MIN(low) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as rsv_20")
        if 'pv_momentum' in factor_list:
            factor_sqls.append("(volume / NULLIF(AVG(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)) * ret as pv_momentum")
        if 'z_score_20' in factor_list:
            factor_sqls.append("(close - AVG(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)) / NULLIF(STDDEV_POP(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as z_score_20")
        if 'buy_vol_ratio' in factor_list:
            factor_sqls.append("SUM(CASE WHEN ret > 0 THEN volume ELSE 0 END) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / NULLIF(SUM(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as buy_vol_ratio")
        if 'kurt_20' in factor_list:
            factor_sqls.append("(AVG(POWER(ret, 4)) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / NULLIF(POWER(STDDEV_POP(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 4), 0)) - 3 as kurt_20")
        if 'rank_pv_corr' in factor_list:
            factor_sqls.append("COVAR_POP(close, volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) / NULLIF(STDDEV_POP(close) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * STDDEV_POP(volume) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) as rank_pv_corr")
        if 'ma_dispersion' in factor_list:
            factor_sqls.append("(ma5 - ma20) / NULLIF(ma20, 0) as ma_dispersion")
        if 'days_to_high' in factor_list:
            factor_sqls.append("1 - (close / NULLIF(MAX(high) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)) as days_to_high")

        # 4. 汇总最终结果
        final_factor_cols = ", ".join(factor_sqls)
        main_sql = f"SELECT date, code, close, {final_factor_cols} FROM base_calc WHERE date >= '{start_date}'"
        
        df = self.db.execute(main_sql).df()
        
        cost = time.time() - start_time
        print(f"[OK] 因子计算成功。耗时: {cost:.2f}s, 结果集大小: {len(df)} 行。")
        return df

if __name__ == "__main__":
    engine = FactorEngine()
    test_df = engine.calculate_factors()
    print(test_df.tail())
