import redis
import psycopg2
import duckdb
import os

class DBConnector:
    """
    量化系统数据库连接器
    统一管理 Redis, QuestDB (Postgres 协议) 和 DuckDB 的连接。
    封装了基础的连接逻辑，确保配置变更时只需修改此处。
    """
    
    def __init__(self, 
                 redis_config={'host': 'localhost', 'port': 6379},
                 questdb_config={'host': 'localhost', 'port': 8812, 'user': 'admin', 'password': 'quest'}):
        
        # 初始化配置信息
        self.redis_config = redis_config
        self.questdb_config = questdb_config
        
    def get_redis(self):
        """
        获取 Redis 客户端实例
        用于：
        1. 缓存静态数据（如股票基本信息）
        2. 任务队列（Celery Broker）
        3. 实时信号发布订阅 (Pub/Sub)
        """
        try:
            r = redis.Redis(**self.redis_config, decode_responses=True)
            # 使用 ping 命令确保服务在运行
            r.ping()
            return r
        except Exception as e:
            print(f"[错误] 无法连接到 Redis 服务: {e}")
            return None

    def get_questdb(self):
        """
        通过 PostgreSQL 协议获取 QuestDB 的连接。
        QuestDB 的 8812 端口支持标准的 Postgres 语法。
        用于：
        1. 执行复杂的时序聚合查询
        2. 导出大规模行情快照
        """
        try:
            # psycopg2 是 Python 访问 Postgres 的标准库
            conn = psycopg2.connect(
                host=self.questdb_config['host'],
                port=self.questdb_config['port'],
                user=self.questdb_config['user'],
                password=self.questdb_config['password'],
                database='qdb' # QuestDB 内部固定的数据库名称
            )
            return conn
        except Exception as e:
            print(f"[错误] 无法连接到 QuestDB 服务: {e}")
            return None

    def get_duckdb_context(self):
        """
        获取 DuckDB 的内存上下文。
        DuckDB 不需要启动后台进程，非常适合作为临时的数据透视分析引擎。
        """
        # 返回一个内存连接，性能最高
        return duckdb.connect(database=':memory:')

if __name__ == "__main__":
    # 模块自测逻辑
    connector = DBConnector()
    print(">>> 正在启动基础设施连接自检...")
    
    redis_conn = connector.get_redis()
    if redis_conn:
        print("✅ Redis 连接就绪")
        
    qdb_conn = connector.get_questdb()
    if qdb_conn:
        print("✅ QuestDB 连接就绪")
        qdb_conn.close()
        
    print("✅ DuckDB 环境就绪")
