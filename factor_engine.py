import pandas as pd
import duckdb
import os
import time
import numpy as np

class FactorEngine:
    def __init__(self):
        self.db = duckdb.connect(database=':memory:') 
        self.data_dir = 'data/daily_k'
        self.index_dir = 'data/index_k'
        self.parquet_glob = os.path.join(self.data_dir, '*.parquet')
        self.index_path = os.path.join(self.index_dir, 'sh.000300.parquet')
        self._is_initialized = False
        
    def _initialize_base_data(self):
        if self._is_initialized: return
        print(">>> 正在初始化内存数据骨架 (单次扫描)...")
        t0 = time.time()
        
        # 1. 加载数据
        self.db.execute(f"CREATE TABLE raw_data AS SELECT * FROM read_parquet('{self.parquet_glob}')")
        self.db.execute(f"CREATE TABLE idx_data AS SELECT date, close as idx_close FROM read_parquet('{self.index_path}')")
        self.db.execute("CREATE TABLE idx_returns AS SELECT date, (idx_close/NULLIF(LAG(idx_close) OVER(ORDER BY date),0)-1) as idx_ret FROM idx_data")
        
        # 2. 构建对齐骨架
        self.db.execute("""
            CREATE TABLE aligned_base AS 
            WITH all_dates AS (SELECT DISTINCT date FROM idx_data),
                 all_codes AS (SELECT DISTINCT code FROM raw_data),
                 skeleton AS (SELECT * FROM all_dates CROSS JOIN all_codes)
            SELECT 
                s.date, s.code,
                LAST_VALUE(r.close IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as close,
                LAST_VALUE(r.open IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as open,
                LAST_VALUE(r.amount IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as amount,
                COALESCE(r.volume, 0) as volume,
                LAG(r.close) OVER (PARTITION BY s.code ORDER BY s.date) as pre_close,
                idx.idx_ret
            FROM skeleton s
            LEFT JOIN raw_data r ON s.date = r.date AND s.code = r.code
            LEFT JOIN idx_returns idx ON s.date = idx.date;
        """)
        self._is_initialized = True
        print(f"[OK] 骨架构建完成。耗时: {time.time()-t0:.2f}s")

    def calculate_all_variants(self, factor_name, windows, start_date='2006-01-01'):
        """
        终极提速：一次扫描计算出所有 Window 组合
        """
        self._initialize_base_data()
        print(f">>> 正在单次扫描计算所有 Window 变体 {windows}...")
        t0 = time.time()
        
        # 预计算通用指标：ret, vwap
        # 使用 CREATE OR REPLACE TABLE 确保 table 物理存在且包含最新列
        self.db.execute("""
            CREATE OR REPLACE TABLE pre_metrics_base AS 
            SELECT *, 
                   (close / NULLIF(LAG(close) OVER(PARTITION BY code ORDER BY date), 0) - 1) as ret,
                   (amount / NULLIF(volume, 0)) as vwap
            FROM aligned_base
        """)
        
        # 构建动态窗口 SQL
        win_sqls = []
        for w in windows:
            w_p = w - 1
            if factor_name == 'smart_money_proxy':
                win_sqls.append(f"((amount/NULLIF(volume,0))/NULLIF(close,0)) as sm_{w}")
            elif factor_name == 'amihud_illiq':
                win_sqls.append(f"AVG((ABS(ret)/NULLIF(amount+1e-9, 0))*1e9) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW) as illiq_{w}")
            elif factor_name == 'skew':
                win_sqls.append(f"((AVG(ret*ret*ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW)) / NULLIF(POWER(STDDEV_POP(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW), 3), 0)) as skew_{w}")
            elif factor_name == 'ivol':
                win_sqls.append(f"STDDEV_POP(ret - idx_ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW) as ivol_{w}")

        sql = f"CREATE OR REPLACE TABLE pre_metrics AS SELECT *, {', '.join(win_sqls)} FROM pre_metrics_base"
        self.db.execute(sql)
        
        res_df = self.db.execute(f"SELECT date, code, open, close, vwap, {', '.join(win_sqls)} FROM pre_metrics WHERE date >= '{start_date}'").df()
        print(f"[OK] 变体批量计算完成。耗时: {time.time()-t0:.2f}s")
        return res_df

    def get_pivoted_factor(self, factor_col, start_date='2006-01-01', factor_name=None, windows=None, fill_na=True):
        """
        利用 DuckDB 强大的 PIVOT 功能，直接在数据库层完成长表转宽表
        """
        self._initialize_base_data()
        
        if factor_name and windows:
             self.calculate_all_variants(factor_name, windows, start_date=start_date)
        elif not self.db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'pre_metrics'").fetchone()[0]:
             self.calculate_all_variants('smart_money_proxy', [10], start_date=start_date)
             
        t0 = time.time()
        sql = f"""
            PIVOT (SELECT date, code, {factor_col} FROM pre_metrics WHERE date >= '{start_date}')
            ON code
            USING FIRST({factor_col})
            GROUP BY date
            ORDER BY date
        """
        df = self.db.execute(sql).df().set_index('date')
        
        # 价格需要填充，但因子不建议填充（代表停牌）
        if fill_na:
            df = df.ffill()
            
        print(f"[OK] 字段 {factor_col} PIVOT 完成 (fill_na={fill_na})。耗时: {time.time()-t0:.2f}s")
        return df

    def get_benchmark_prices(self, start_date='2006-01-01'):
        """
        获取基准指数 (沪深300) 的价格序列
        """
        self._initialize_base_data()
        df = self.db.execute(f"SELECT date, idx_close FROM idx_data WHERE date >= '{start_date}' ORDER BY date").df()
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['idx_close']
