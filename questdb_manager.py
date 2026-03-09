import pandas as pd
import os
import psycopg2
import time
from db_connector import DBConnector
from tqdm import tqdm

class QuestDBManager:
    """
    QuestDB 实战管理器
    负责时序数据的 Schema 管理、高性能批量灌单以及时序聚合分析。
    【为什么用 QuestDB？】
    它是一款专门为时序设计的数据库，处理 TB 级行情数据依然能保持极高的查询性能，
    特别是其独特的 SAMPLE BY 语法（重采样）。
    """
    
    def __init__(self):
        # 依赖之前定义的 DBConnector 获取连接信息
        self.connector = DBConnector()
        # 采用 Postgres 协议进行交互
        self.conn = self.connector.get_questdb()
        if self.conn:
            # QuestDB 的 DDL 语句通常需要开启 autocommit
            self.conn.autocommit = True
        
    def init_tables(self):
        """
        在数据库中初始化量化核心表结构。
        技术亮点：
        1. 使用 SYMBOL 类型：对股票代码进行索引优化，极大节省空间并加速过滤查询。
        2. WAL 模式：Write-Ahead Logging 模式，极大提升并发写入能力。
        3. PARTITION BY YEAR：按年进行物理分区，是时序数据库提升历史查询速度的关键。
        """
        if not self.conn:
            print("[Error] 无法连接到 QuestDB，请确认 8812 端口是否开启。")
            return
        
        with self.conn.cursor() as cur:
            print(">>> 正在初始化 QuestDB 表结构 (WAL 模式)...")
            
            # 1. 创建个股日线表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_daily (
                    timestamp TIMESTAMP,
                    code SYMBOL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    preclose DOUBLE,
                    volume DOUBLE,
                    amount DOUBLE,
                    turn DOUBLE,
                    tradestatus STRING,
                    pctChg DOUBLE,
                    peTTM DOUBLE,
                    psTTM DOUBLE,
                    pcfNcfTTM DOUBLE,
                    pbMRQ DOUBLE,
                    isST STRING
                ) timestamp(timestamp) PARTITION BY YEAR WAL;
            """)
            
            # 2. 创建指数日线表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS index_daily (
                    timestamp TIMESTAMP,
                    code SYMBOL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    preclose DOUBLE,
                    volume DOUBLE,
                    amount DOUBLE,
                    pctChg DOUBLE
                ) timestamp(timestamp) PARTITION BY YEAR WAL;
            """)
            print("[OK] 表结构部署成功。")

    def _get_last_sync_timestamp(self, table, code):
        """查询数据库，获取该标的已经存在的最新时间点，实现增量同步"""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT max(timestamp) FROM {table} WHERE code = '{code}'")
            res = cur.fetchone()
            return res[0] if res and res[0] else None

    def sync_parquet_to_questdb(self, data_dir="data"):
        """
        增量同步：将本地 Parquet 文件（数据湖）同步入库。
        只处理那些数据库中还没有的新记录。
        """
        if not self.conn: return
        
        # 待同步的任务列表
        sync_tasks = [
            {'table': 'stock_daily', 'dir': os.path.join(data_dir, "daily_k")},
            {'table': 'index_daily', 'dir': os.path.join(data_dir, "index_k")}
        ]

        for task in sync_tasks:
            target_table = task['table']
            source_dir = task['dir']
            
            if not os.path.exists(source_dir):
                print(f"[Warning] 路径不存在，跳过: {source_dir}")
                continue
            
            files = [f for f in os.listdir(source_dir) if f.endswith('.parquet')]
            print(f"\n>>> 正在同步 {source_dir} -> QuestDB [{target_table}]...")

            for file in tqdm(files, desc="入库中"):
                code = file.replace('.parquet', '')
                df = pd.read_parquet(os.path.join(source_dir, file))
                
                # 时间轴转换：QuestDB 识别 timestamp 列作为主轴
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop(columns=['date'])
                
                # 增量逻辑：只取本地比数据库更晚的数据
                last_ts = self._get_last_sync_timestamp(target_table, code)
                if last_ts:
                    df = df[df['timestamp'] > last_ts]
                
                if df.empty:
                    continue

                # 执行批量高效写入
                self._execute_batch_insert(df, target_table)

    def _execute_batch_insert(self, df, table):
        """
        高性能批量写入函数。
        利用多行插入 (Bulk Insert) 显著减少网络开销和磁盘 I/O。
        """
        cols = ",".join(df.columns)
        values_list = df.values.tolist()
        
        with self.conn.cursor() as cur:
            # 每 2000 行为一批，兼顾内存占用和入库效率
            chunk_size = 2000
            for i in range(0, len(values_list), chunk_size):
                chunk = values_list[i : i + chunk_size]
                # 构造 SQL 占位符 (%s, %s, ...)
                placeholders = ",".join(["(" + ",".join(["%s"] * len(df.columns)) + ")"] * len(chunk))
                # 展平列表适配 psycopg2 的 execute
                flattened_values = [item for sublist in chunk for item in sublist]
                
                insert_query = f"INSERT INTO {table} ({cols}) VALUES {placeholders}"
                cur.execute(insert_query, flattened_values)

    def query_example_resampling(self):
        """
        展示 QuestDB 的杀手锏：一键生成季度 K 线 (SAMPLE BY)。
        这是量化中进行多周期分析最强大的功能。
        """
        print("\n>>> 展示 SAMPLE BY 功能：生成 sh.600519 季度行情...")
        sql = """
            SELECT 
                timestamp, 
                code, 
                first(open) as open, 
                max(high) as high, 
                min(low) as low, 
                last(close) as close, 
                sum(volume) as total_volume
            FROM stock_daily 
            WHERE code = 'sh.600519'
            SAMPLE BY 3M ALIGN TO CALENDAR; -- 按 3 个月（季度）重新采样
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            df_res = pd.DataFrame(rows, columns=['timestamp', 'code', 'open', 'high', 'low', 'close', 'vol'])
            print(df_res.tail())

if __name__ == "__main__":
    qdb_mgr = QuestDBManager()
    
    # 1. 初始化表
    qdb_mgr.init_tables()
    
    # 2. 从本地 Parquet 入库
    start_sync = time.time()
    qdb_mgr.sync_parquet_to_questdb()
    print(f"\n[OK] 同步完成，总耗时: {time.time() - start_sync:.2f}s")
    
    # 3. 演示一个查询
    qdb_mgr.query_example_resampling()
