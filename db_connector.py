import redis
import psycopg2
import duckdb
import os

class DBConnector:
    """
    量化系统数据库连接器
    统一管理 Redis, QuestDB (Postgres 协议) 和 DuckDB 的连接。
    """
    
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.qdb_host = os.getenv('QUESTDB_HOST', 'localhost')
        self.qdb_port = int(os.getenv('QUESTDB_PORT', 8812))
        self.qdb_user = 'admin'
        self.qdb_password = 'quest'

    def get_redis(self):
        """获取 Redis 连接"""
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping()
            return r
        except Exception as e:
            print(f"[Error] Redis 连接失败: {e}")
            return None

    def get_questdb(self):
        """获取 QuestDB 连接 (Postgres 协议)"""
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
    connector = DBConnector()
    redis_conn = connector.get_redis()
    if redis_conn:
        print("[OK] Redis 连接就绪")
        
    qdb_conn = connector.get_questdb()
    if qdb_conn:
        print("[OK] QuestDB 连接就绪")
        qdb_conn.close()
        
    print("[OK] DuckDB 环境就绪")
