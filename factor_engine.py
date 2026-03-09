import pandas as pd
import duckdb
import os
import time
import numpy as np

class FactorEngine:
    """
    极速因子计算引擎
    基于 DuckDB (嵌入式 OLAP 数据库) 实现，专为大规模金融因子计算而生。
    【为什么用 DuckDB？】
    1. 计算性能极强：SQL 窗口函数天然支持并行计算，比 Pandas 循环快得多。
    2. 内存管理：虽然在内存中运行，但支持高效的列式存储和谓词下推。
    3. 直接读取 Parquet：无需经过内存转换，直接在硬盘数据上执行 SQL。
    """
    def __init__(self):
        # 创建一个内存中的 DuckDB 数据库实例
        self.db = duckdb.connect(database=':memory:') 
        self.data_dir = 'data/daily_k'
        self.index_dir = 'data/index_k'
        # 匹配该目录下所有的个股 Parquet 文件
        self.parquet_glob = os.path.join(self.data_dir, '*.parquet')
        # 基准指数路径
        self.index_path = os.path.join(self.index_dir, 'sh.000300.parquet')
        self._is_initialized = False
        
    def _initialize_base_data(self):
        """
        初始化内存数据“骨架”。
        解决量化痛点：不同股票上市日期不同、存在停牌，导致数据无法直接对齐进行矩阵运算。
        解决方案：
        1. 获取全量交易日和全量股票代码。
        2. 通过 CROSS JOIN 生成全日期-全代码的“格子”。
        3. 用 LAST_VALUE(...) IGNORE NULLS 实现停牌期间的价格前向填充 (Forward Fill)。
        """
        if self._is_initialized: return
        print(">>> 正在初始化内存数据骨架 (单次全量扫描)...")
        t0 = time.time()
        
        # 1. 扫描磁盘上的所有 Parquet 文件并加载进内存
        self.db.execute(f"CREATE TABLE raw_data AS SELECT * FROM read_parquet('{self.parquet_glob}')")
        self.db.execute(f"CREATE TABLE idx_data AS SELECT date, close as idx_close FROM read_parquet('{self.index_path}')")
        # 预计算指数收益率，用于计算超额收益或特异性风险因子
        self.db.execute("CREATE TABLE idx_returns AS SELECT date, (idx_close/NULLIF(LAG(idx_close) OVER(ORDER BY date),0)-1) as idx_ret FROM idx_data")
        
        # 2. 构建对齐后的基础宽表
        self.db.execute("""
            CREATE TABLE aligned_base AS 
            WITH all_dates AS (SELECT DISTINCT date FROM idx_data),
                 all_codes AS (SELECT DISTINCT code FROM raw_data),
                 skeleton AS (SELECT * FROM all_dates CROSS JOIN all_codes)
            SELECT 
                s.date, s.code,
                -- 核心逻辑：如果当天停牌没有数据，就取最近的一个非空成交价
                LAST_VALUE(r.close IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as close,
                LAST_VALUE(r.open IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as open,
                LAST_VALUE(r.amount IGNORE NULLS) OVER (PARTITION BY s.code ORDER BY s.date) as amount,
                COALESCE(r.volume, 0) as volume, -- 停牌天成交量显式设为 0
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
        一次性批量计算因子的多种时间窗口变体。
        比如：同时计算 10日和20日的聪明钱因子，避免重复扫描数据。
        """
        self._initialize_base_data()
        print(f">>> 正在单次扫描计算所有 Window 变体 {windows}...")
        t0 = time.time()
        
        # 预计算通用基础指标：收益率(ret)和成交均价(vwap)
        self.db.execute("""
            CREATE OR REPLACE TABLE pre_metrics_base AS 
            SELECT *, 
                   (close / NULLIF(LAG(close) OVER(PARTITION BY code ORDER BY date), 0) - 1) as ret,
                   (amount / NULLIF(volume, 0)) as vwap
            FROM aligned_base
        """)
        
        # 动态构建 SQL 语句块，处理不同的因子计算逻辑
        win_sqls = []
        for w in windows:
            w_p = w - 1
            if factor_name == 'smart_money_proxy':
                # 示例因子：(成交均价 / 收盘价) 的移动平均
                win_sqls.append(f"AVG((amount/NULLIF(volume,0))/NULLIF(close,0)) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW) as sm_{w}")
            elif factor_name == 'amihud_illiq':
                # 非流动性因子：|收益率| / 成交额
                win_sqls.append(f"AVG((ABS(ret)/NULLIF(amount+1e-9, 0))*1e9) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW) as illiq_{w}")
            elif factor_name == 'skew':
                # 收益率偏度因子
                win_sqls.append(f"((AVG(ret*ret*ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW)) / NULLIF(POWER(STDDEV_POP(ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW), 3), 0)) as skew_{w}")
            elif factor_name == 'ivol':
                # 特异性风险因子 (IVOL)：个股相对于指数收益率的残差标准差
                win_sqls.append(f"STDDEV_POP(ret - idx_ret) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN {w_p} PRECEDING AND CURRENT ROW) as ivol_{w}")

        # 执行因子计算 SQL
        sql = f"CREATE OR REPLACE TABLE pre_metrics AS SELECT *, {', '.join(win_sqls)} FROM pre_metrics_base"
        self.db.execute(sql)
        
        # 最终以 DataFrame 形式提取数据
        res_df = self.db.execute(f"SELECT date, code, open, close, vwap, {', '.join(win_sqls)} FROM pre_metrics WHERE date >= '{start_date}'").df()
        print(f"[OK] 变体批量计算完成。耗时: {time.time()-t0:.2f}s")
        return res_df

    def get_pivoted_factor(self, factor_col, start_date='2006-01-01', factor_name=None, windows=None, fill_na=True):
        """
        利用 DuckDB 极速 PIVOT 功能，将“长表”转为“宽表矩阵” (Index: Date, Columns: Stocks)。
        这是矩阵运算（如排名、回测）最喜欢的格式。
        """
        self._initialize_base_data()
        
        # 检查是否需要计算因子
        if factor_name and windows:
             self.calculate_all_variants(factor_name, windows, start_date=start_date)
        elif not self.db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'pre_metrics'").fetchone()[0]:
             self.calculate_all_variants('smart_money_proxy', [10], start_date=start_date)
             
        t0 = time.time()
        # 原生 SQL PIVOT
        sql = f"""
            PIVOT (SELECT date, code, {factor_col} FROM pre_metrics WHERE date >= '{start_date}')
            ON code
            USING FIRST({factor_col})
            GROUP BY date
            ORDER BY date
        """
        df = self.db.execute(sql).df().set_index('date')
        
        # 价格类数据需要前向填充（代表持仓维持不变），但因子类通常不填充
        if fill_na:
            df = df.ffill()
            
        print(f"[OK] 字段 {factor_col} PIVOT 完成 (fill_na={fill_na})。耗时: {time.time()-t0:.2f}s")
        return df

    def get_benchmark_prices(self, start_date='2006-01-01'):
        """获取对照组指数 (如沪深300) 的价格序列"""
        self._initialize_base_data()
        df = self.db.execute(f"SELECT date, idx_close FROM idx_data WHERE date >= '{start_date}' ORDER BY date").df()
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['idx_close']
