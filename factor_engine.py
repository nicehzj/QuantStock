import duckdb
import pandas as pd
import os
import time
from db_connector import DBConnector

class FactorEngine:
    """
    因子计算引擎
    """
    
    def __init__(self, data_source='parquet'):
        self.data_source = data_source
        self.connector = DBConnector()
        self.db = duckdb.connect(database=':memory:')
        self.data_path = "data/daily_k"
        self.parquet_glob = os.path.join(self.data_path, "*.parquet")

    def _prepare_data_source(self):
        if self.data_source == 'questdb':
            qdb_conn = self.connector.get_questdb()
            if not qdb_conn:
                raise ConnectionError("无法连接到 QuestDB。请确保 QuestDB 正在运行且 8812 端口可访问。")
            
            print(">>> 正在从 QuestDB 提取行情数据...")
            query = "SELECT timestamp as date, code, close, volume, pctChg FROM stock_daily"
            raw_df = pd.read_sql(query, qdb_conn)
            qdb_conn.close()
            
            if raw_df.empty:
                raise ValueError("QuestDB 中没有任何行情数据，请先运行 questdb_manager.py 进行同步。")
            
            self.db.register('raw_market_data', raw_df)
            return "raw_market_data"
        
        else:
            if not os.path.exists(self.data_path) or not os.listdir(self.data_path):
                raise FileNotFoundError(f"Parquet 数据湖为空: {self.data_path}")
            return f"read_parquet('{self.parquet_glob}')"

    def calculate_basic_factors(self):
        start_time = time.time()
        source_from = self._prepare_data_source()
        print(f">>> 计算引擎启动 [模式: {self.data_source}]...")
        
        sql = f"""
        WITH base_data AS (
            SELECT CAST(date AS DATE) as date, code, close, volume, pctChg, isST
            FROM {source_from}
            WHERE isST != '1' 
        ),
        factor_calc AS (
            SELECT 
                date, code, close,
                (close / LAG(close, 20) OVER(PARTITION BY code ORDER BY date) - 1) as factor_mom_20,
                (close / AVG(close) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) - 1) as factor_ma_gap,
                STDDEV(pctChg) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * SQRT(252) as factor_vol_20
            FROM base_data
        )
        SELECT * FROM factor_calc 
        WHERE factor_mom_20 IS NOT NULL 
        ORDER BY date, code
        """
        result_df = self.db.execute(sql).df()
        duration = time.time() - start_time
        print(f"[OK] 因子计算成功。数据源: {self.data_source}, 耗时: {duration:.2f}s, 结果集大小: {len(result_df)} 行。")
        return result_df

    def get_latest_market_snapshot(self, factor_df):
        latest_date = factor_df['date'].max()
        snapshot = factor_df[factor_df['date'] == latest_date].copy()
        return latest_date, snapshot

if __name__ == "__main__":
    try:
        engine = FactorEngine(data_source='questdb')
        factors = engine.calculate_basic_factors()
        print("\n--- 因子计算结果展示 ---")
        print(factors.tail(10))
        ldate, snap = engine.get_latest_market_snapshot(factors)
        top_picks = snap.sort_values('factor_mom_20', ascending=False).head(5)
        print(f"\n--- {ldate} 动量因子 TOP 5 ---")
        print(top_picks[['code', 'close', 'factor_mom_20']])
    except Exception as e:
        print(f"[Error] 执行失败: {e}")
        print("[Tip] 提示: 请确保 QuestDB 已同步数据且端口 8812 已开放。")
