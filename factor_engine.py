import duckdb
import pandas as pd
import os
import time
from db_connector import DBConnector

class FactorEngine:
    """
    因子计算引擎 (v2.0 - 混合数据源支持)
    利用 DuckDB 的 OLAP 能力，在 Parquet 数据湖或 QuestDB 时序库上进行因子挖掘。
    核心优势：计算逻辑与存储介质解耦。
    """
    
    def __init__(self, data_source='parquet'):
        """
        :param data_source: 
            'parquet' - 直接读取磁盘上的 .parquet 文件（适合大规模离线回测）
            'questdb' - 从 QuestDB 时序数据库读取数据（适合增量计算和模拟实盘）
        """
        self.data_source = data_source
        self.connector = DBConnector()
        # 初始化内存级 DuckDB 连接
        self.db = duckdb.connect(database=':memory:')
        
        # 本地 Parquet 路径配置
        self.data_path = "data/daily_k"
        self.parquet_glob = os.path.join(self.data_path, "*.parquet")

    def _prepare_data_source(self):
        """
        内部逻辑：准备数据源。
        如果是 QuestDB，将数据提取到内存并注册到 DuckDB。
        如果是 Parquet，返回 DuckDB 的 read_parquet 路径。
        """
        if self.data_source == 'questdb':
            # 1. 从 QuestDB 获取数据
            qdb_conn = self.connector.get_questdb()
            if not qdb_conn:
                raise ConnectionError("无法连接到 QuestDB。请确保 QuestDB 正在运行且 8812 端口可访问。")
            
            print(">>> 正在从 QuestDB 提取行情数据...")
            # 这里的 SQL 可以利用 QuestDB 的特性进行预过滤
            query = "SELECT timestamp as date, code, close, volume, pctChg FROM stock_daily"
            
            # 使用 Pandas 作为高速数据交换的中转
            raw_df = pd.read_sql(query, qdb_conn)
            qdb_conn.close()
            
            if raw_df.empty:
                raise ValueError("QuestDB 中没有任何行情数据，请先运行 questdb_manager.py 进行同步。")
            
            # 2. 将 DataFrame 注册到 DuckDB 的虚拟表空间中
            self.db.register('raw_market_data', raw_df)
            return "raw_market_data"
        
        else:
            # Parquet 模式：检查目录
            if not os.path.exists(self.data_path) or not os.listdir(self.data_path):
                raise FileNotFoundError(f"Parquet 数据湖为空: {self.data_path}")
            return f"read_parquet('{self.parquet_glob}')"

    def calculate_basic_factors(self):
        """
        批量计算全市场因子。
        计算逻辑通过标准 SQL 窗口函数实现。
        """
        start_time = time.time()
        
        # 确定底层数据表/路径
        source_from = self._prepare_data_source()
        
        print(f">>> 计算引擎启动 [模式: {self.data_source}]...")
        
        # 核心因子 SQL
        sql = f"""
        WITH base_data AS (
            SELECT 
                CAST(date AS DATE) as date,
                code,
                close,
                volume,
                pctChg,
                isST
            FROM {source_from}
            WHERE isST != '1' -- 核心修复：剔除 ST 股
        ),
        factor_calc AS (
            SELECT 
                date,
                code,
                close,
                -- A. 20日收益率动量
                (close / LAG(close, 20) OVER(PARTITION BY code ORDER BY date) - 1) as factor_mom_20,
                
                -- B. 价格相对于 20 日均线的偏离度 (均值回归因子)
                (close / AVG(close) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) - 1) as factor_ma_gap,
                
                -- C. 20日年化收益波动率
                STDDEV(pctChg) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * SQRT(252) as factor_vol_20
            FROM base_data
        )
        SELECT * FROM factor_calc 
        WHERE factor_mom_20 IS NOT NULL 
        ORDER BY date, code
        """
        
        # 执行计算
        result_df = self.db.execute(sql).df()
        
        duration = time.time() - start_time
        print(f"✅ 因子计算成功。数据源: {self.data_source}, 耗时: {duration:.2f}s, 结果集大小: {len(result_df)} 行。")
        
        return result_df

    def get_latest_market_snapshot(self, factor_df):
        """
        获取全市场最新一个交易日的因子截面数据。
        """
        latest_date = factor_df['date'].max()
        snapshot = factor_df[factor_df['date'] == latest_date].copy()
        return latest_date, snapshot

if __name__ == "__main__":
    # --- 演示：体验数据库驱动的计算 ---
    try:
        # 您可以在此处尝试修改为 'parquet' 进行对比测试
        engine = FactorEngine(data_source='questdb')
        
        factors = engine.calculate_basic_factors()
        
        print("\n--- 因子计算结果展示 ---")
        print(factors.tail(10))
        
        # 简单选股：查看最新动量最强的股票
        ldate, snap = engine.get_latest_market_snapshot(factors)
        top_picks = snap.sort_values('factor_mom_20', ascending=False).head(5)
        print(f"\n--- {ldate} 动量因子 TOP 5 ---")
        print(top_picks[['code', 'close', 'factor_mom_20']])
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        print("💡 提示: 请确保 QuestDB 已同步数据且端口 8812 已开放。")
