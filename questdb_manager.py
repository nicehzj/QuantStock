import pandas as pd
import os
import psycopg2
import time
from db_connector import DBConnector
from tqdm import tqdm

class QuestDBManager:
    """
    QuestDB 实战管理器
    负责时序数据的 Schema 管理、高性能批量灌单以及特有的时序分析查询。
    """
    
    def __init__(self):
        # 依赖之前定义的 DBConnector 获取连接信息
        self.connector = DBConnector()
        # 采用 PostgreSQL 协议进行交互
        self.conn = self.connector.get_questdb()
        if self.conn:
            # 开启自动提交，QuestDB 的 DDL 语句通常需要此设置
            self.conn.autocommit = True
        
    def init_tables(self):
        """
        在 QuestDB 中初始化量化核心表结构。
        使用 SYMBOL 类型优化股票代码存储，使用 WAL 模式增强并发。
        """
        if not self.conn:
            print("❌ 无法连接到 QuestDB，请检查 8812 端口是否开启。")
            return
        
        with self.conn.cursor() as cur:
            print(">>> 正在 QuestDB 中部署量化表结构 (WAL 模式)...")
            
            # 1. 个股日线表：包含全部 16 个字段 + 指定时间轴 timestamp
            # PARTITION BY YEAR: 历史日线数据按年分区即可，既保证性能又方便管理
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
            
            # 2. 指数日线表
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
            print("✅ 表结构部署成功。")

    def _get_last_sync_timestamp(self, table, code):
        """查询 QuestDB，获取单只股票已入库的最晚时间戳"""
        with self.conn.cursor() as cur:
            # QuestDB 优化的过滤语法
            cur.execute(f"SELECT max(timestamp) FROM {table} WHERE code = '{code}'")
            res = cur.fetchone()
            return res[0] if res and res[0] else None

    def sync_parquet_to_questdb(self, data_dir="data"):
        """
        将本地 Parquet 数据湖同步至 QuestDB。
        采用增量同步机制：只读取比 QuestDB 中更新的记录。
        """
        if not self.conn: return
        
        # 配置同步任务路径
        sync_tasks = [
            {'table': 'stock_daily', 'dir': os.path.join(data_dir, "daily_k")},
            {'table': 'index_daily', 'dir': os.path.join(data_dir, "index_k")}
        ]

        for task in sync_tasks:
            target_table = task['table']
            source_dir = task['dir']
            
            if not os.path.exists(source_dir):
                print(f"⚠️ 路径不存在，跳过: {source_dir}")
                continue
            
            files = [f for f in os.listdir(source_dir) if f.endswith('.parquet')]
            print(f"\n>>> 正在将 {source_dir} 同步至 QuestDB [{target_table}]...")

            for file in tqdm(files, desc="入库进度"):
                code = file.replace('.parquet', '')
                df = pd.read_parquet(os.path.join(source_dir, file))
                
                # 时间轴转换：QuestDB 识别 timestamp 列作为主轴
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop(columns=['date'])
                
                # 增量判断
                last_ts = self._get_last_sync_timestamp(target_table, code)
                if last_ts:
                    df = df[df['timestamp'] > last_ts]
                
                if df.empty:
                    continue

                # 执行批量插入
                self._execute_batch_insert(df, target_table)

    def _execute_batch_insert(self, df, table):
        """
        利用多行插入提高写入效率。
        在大规模（千万级）写入时，QuestDB 推荐使用 ILP 协议，
        此处作为日线级别同步，Postgres 批量插入已足够快。
        """
        cols = ",".join(df.columns)
        values_list = df.values.tolist()
        
        with self.conn.cursor() as cur:
            # 每 2000 行执行一次批量插入，平衡内存与速度
            chunk_size = 2000
            for i in range(0, len(values_list), chunk_size):
                chunk = values_list[i : i + chunk_size]
                # 构造占位符 (%s, %s, ...)
                placeholders = ",".join(["(" + ",".join(["%s"] * len(df.columns)) + ")"] * len(chunk))
                # 展平嵌套列表以适配 cur.execute
                flattened_values = [item for sublist in chunk for item in sublist]
                
                insert_query = f"INSERT INTO {table} ({cols}) VALUES {placeholders}"
                cur.execute(insert_query, flattened_values)

    def query_example_resampling(self):
        """
        展示 QuestDB 的时序采样 (Resampling) 威力。
        例如：将日线数据聚合为季度线。
        """
        print("\n>>> QuestDB 威力展示：一键生成季度 K 线 (SAMPLE BY)...")
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
            SAMPLE BY 3M ALIGN TO CALENDAR;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            df_res = pd.DataFrame(rows, columns=['timestamp', 'code', 'open', 'high', 'low', 'close', 'vol'])
            print(df_res.tail())

if __name__ == "__main__":
    qdb_mgr = QuestDBManager()
    
    # 步骤 1: 建表
    qdb_mgr.init_tables()
    
    # 步骤 2: 数据从湖（Parquet）入库（QuestDB）
    start_sync = time.time()
    qdb_mgr.sync_parquet_to_questdb()
    print(f"\n✨ 同步完成，耗时: {time.time() - start_sync:.2f}s")
    
    # 步骤 3: 运行一个时序分析查询
    qdb_mgr.query_example_resampling()
