import redis
import psycopg2
import duckdb
import os

class DBConnector:
    """
    量化系统数据库连接器
    统一管理并分发三种关键数据库的连接：
    1. Redis: 用于存放实时交易信号、缓存计算结果、任务队列。
    2. QuestDB: 高性能时序数据库，用于持久化存储海量 K 线和 Tick 数据。
    3. DuckDB: 嵌入式分析型数据库，负责在内存中进行超高速的因子计算。
    """
    
    def __init__(self):
        # 默认数据库配置，优先从环境变量读取
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.qdb_host = os.getenv('QUESTDB_HOST', 'localhost')
        self.qdb_port = int(os.getenv('QUESTDB_PORT', 8812)) # QuestDB 的 Postgres 协议端口
        self.qdb_user = 'admin'
        self.qdb_password = 'quest'

    def get_redis(self):
        """
        获取 Redis 连接。
        量化用途：比如保存今天建议买入的股票列表，或者缓存已经计算好的因子矩阵。
        """
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping() # 测试连接是否连通
            return r
        except Exception as e:
            print(f"[Error] Redis 连接失败: {e}")
            return None

    def get_questdb(self):
        """
        获取 QuestDB 连接 (基于 Postgres 协议)。
        QuestDB 虽然有自己的存储内核，但对外兼容 Postgres 生态，非常方便集成。
        """
        try:
            conn = psycopg2.connect(
                host=self.qdb_host,
                port=self.qdb_port,
                user=self.qdb_user,
                password=self.qdb_password,
                database='qdb'
            )
            return conn
        except Exception as e:
            print(f"[Error] QuestDB 连接失败: {e}")
            return None

if __name__ == "__main__":
    # 单元测试：检查本地数据库环境
    connector = DBConnector()
    redis_conn = connector.get_redis()
    if redis_conn:
        print("[OK] Redis 连接就绪")
        
    qdb_conn = connector.get_questdb()
    if qdb_conn:
        print("[OK] QuestDB 连接就绪")
        qdb_conn.close()
        
    print("[OK] DuckDB 环境就绪 (DuckDB 通常直接在内存中创建连接)")
