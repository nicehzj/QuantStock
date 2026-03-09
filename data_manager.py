import baostock as bs
import pandas as pd
import os
import json
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

# =====================================================================
# 🛠️ 子进程初始化逻辑
# Baostock 库限制：不支持多线程并发，但支持多进程。
# 每一个子进程都必须拥有独立的登录会话 (bs.login()) 才能正常获取数据。
# =====================================================================
_child_initialized = False

def init_worker():
    """
    每个子进程启动时都会调用的初始化函数。
    它确保子进程在执行抓取任务前已经成功登录了 Baostock。
    """
    global _child_initialized
    if not _child_initialized:
        bs.login()
        _child_initialized = True

def sync_worker(task_info):
    """
    单个股票或指数的同步核心函数（在子进程中运行）。
    参数 task_info 包含：(代码, 是否是指数, 开始日期, 结束日期, 保存目录)。
    """
    code, is_index, start_date, end_date, save_dir = task_info
    if start_date > end_date:
        return code, None

    # 定义数据字段：
    # 指数仅需量价；个股则额外需要换手率(turn)、市盈率(peTTM)、是否ST等基本面因子。
    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg" if is_index else \
             "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    
    # 复权说明：指数不复权(3)，个股统一使用【后复权(1)】。
    # 后复权可以保证历史收益率计算的准确性，且不会因为除权息导致价格突变。
    adjustflag = "3" if is_index else "1"

    data_rows = []
    success = False
    
    try:
        # 异常重试机制：网络不稳定时自动尝试最多3次
        for i in range(3):
            rs = bs.query_history_k_data_plus(
                code, fields,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adjustflag
            )
            if rs.error_code == '0':
                while rs.next():
                    data_rows.append(rs.get_row_data())
                success = True
                break
            else:
                # 随机等待 0-2 秒，防止高频请求被服务器拒绝
                time.sleep(random.random() * 2)
    except:
        pass

    if not success or not data_rows:
        # 如果是老股票（已退市）获取不到最新数据，返回 end_date 标记为已同步。
        return code, end_date

    # 将结果转换为 Pandas DataFrame 并处理数值类型
    new_df = pd.DataFrame(data_rows, columns=rs.fields)
    numeric_cols = [c for c in new_df.columns if c not in ['date', 'code', 'tradestatus', 'isST']]
    new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 持久化存储：使用 Parquet 格式（带 Snappy 压缩）。
    # Parquet 是列式存储，读取速度极快，是大数据和量化分析的首选格式。
    file_path = os.path.join(save_dir, f"{code}.parquet")
    if os.path.exists(file_path):
        # 增量合并：读取本地已有数据 -> 拼接新数据 -> 按日期去重排序
        existing_df = pd.read_parquet(file_path)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date']).sort_values('date')
        combined_df.to_parquet(file_path, index=False, compression='snappy')
        max_date = combined_df['date'].max()
    else:
        new_df.to_parquet(file_path, index=False, compression='snappy')
        max_date = new_df['date'].max()
    
    return code, max_date

class BaostockDataManager:
    """
    Baostock 数据中心管理器
    负责自动化管理沪深 A 股全量历史数据的抓取、本地存储（数据湖）及增量更新。
    """
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.stock_path = os.path.join(base_path, "daily_k")
        self.index_path = os.path.join(base_path, "index_k")
        self.pool_cache_path = os.path.join(base_path, "stock_pools")
        self.metadata_file = os.path.join(base_path, "sync_metadata.json") # 同步进度记录文件
        
        # 自动创建必要的文件夹
        os.makedirs(self.stock_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.pool_cache_path, exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """加载本地同步记录，确保程序重启后能从上次停下的地方继续"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"stocks": {}, "indexes": {}}

    def _save_metadata(self):
        """将当前每只股的同步进度保存到 JSON 文件"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def _get_effective_date(self):
        """
        确定今日同步的截止日期。
        量化常识：A 股数据通常在收盘后（晚上 8 点左右）才会完全结清当日最终行情。
        """
        bs.login()
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        rs = bs.query_trade_dates(start_date=(now - timedelta(days=10)).strftime("%Y-%m-%d"), end_date=today_str)
        df = rs.get_data()
        bs.logout()
        if df.empty: return today_str
        trading_days = df[df['is_trading_day'] == '1']['calendar_date'].tolist()
        if not trading_days: return today_str
        last_trading_day = trading_days[-1]
        
        # 如果今天是交易日但还没到晚 8 点，为了保证数据完整，同步到前一个交易日。
        if last_trading_day == today_str and now.hour < 20:
            return trading_days[-2] if len(trading_days) >= 2 else last_trading_day
        return last_trading_day

    def get_master_stock_pool(self, start_year=2006):
        """
        核心优化：获取从 2006 年至今每年初在市股票的并集，彻底解决【幸存者偏差】。
        【什么是幸存者偏差？】
        如果你只用现在的股票名单去回测 2010 年，你就漏掉了当时存在但后来退市的股票，导致回测结果虚高。
        本系统通过遍历历史每一年的全量名单并取并集，来真实还原历史市场环境。
        """
        master_pool_file = os.path.join(self.pool_cache_path, "master_pool.json")
        
        # 1. 优先尝试加载本地缓存的总名单
        if os.path.exists(master_pool_file):
            with open(master_pool_file, 'r', encoding='utf-8') as f:
                cached_pool = json.load(f)
                print(f"[OK] 从本地加载全局股票池，共计 {len(cached_pool)} 只股票。")
                return cached_pool

        print(f">>> 正在构建从 {start_year} 年至今的全局股票池...")
        current_year = datetime.now().year
        master_pool = set()
        
        bs.login()
        for year in range(start_year, current_year + 1):
            year_cache_file = os.path.join(self.pool_cache_path, f"stock_pool_{year}.json")
            
            # 如果该年份的名单已下载过，直接加载
            if os.path.exists(year_cache_file):
                with open(year_cache_file, 'r', encoding='utf-8') as f:
                    year_stocks = json.load(f)
                    master_pool.update(year_stocks)
                    print(f"  - {year} 年数据: 已从本地加载 ({len(year_stocks)} 只)")
                    continue

            # 获取该年年初的第一个交易日
            target_date = f"{year}-01-01"
            rs_date = bs.query_trade_dates(start_date=target_date, end_date=f"{year}-01-20")
            df_date = rs_date.get_data()
            if not df_date.empty:
                valid_days = df_date[df_date['is_trading_day'] == '1']
                if not valid_days.empty:
                    first_day = valid_days.iloc[0]['calendar_date']
                    # 查询该交易日的全市场股票列表
                    rs = bs.query_all_stock(day=first_day)
                    df = rs.get_data()
                    if not df.empty:
                        # 过滤出沪深两市的主板、创业板、科创板股票
                        stocks = df[df['code'].str.contains(r'^sz\.[0-3]|^sh\.6')]['code'].tolist()
                        master_pool.update(stocks)
                        
                        # 缓存该年名单
                        with open(year_cache_file, 'w', encoding='utf-8') as f:
                            json.dump(stocks, f, indent=4)
                        print(f"  - {year} 年数据: 已从 Baostock 获取并缓存 ({len(stocks)} 只)")
        
        bs.logout()
        sorted_pool = sorted(list(master_pool))
        
        # 保存最终合并的大池子
        with open(master_pool_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_pool, f, indent=4)
            
        print(f"[OK] 全局股票池构建完成，共计 {len(sorted_pool)} 只股票 (包含已退市)。")
        return sorted_pool

    def run_sync(self, codes, is_index=False, max_workers=4):
        """
        启动高并发同步引擎。
        利用多核 CPU 分布式抓取数据，极大缩短同步时间。
        """
        label = "指数" if is_index else "个股"
        meta_type = "indexes" if is_index else "stocks"
        save_dir = self.index_path if is_index else self.stock_path
        end_date = self._get_effective_date()
        
        tasks = []
        for code in codes:
            # 确定同步起始日期（上次截止日期 + 1天）
            last = self.metadata[meta_type].get(code, "2006-01-01")
            start = last if last == "2006-01-01" else (pd.to_datetime(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((code, is_index, start, end_date, save_dir))

        print(f">>> 启动多进程同步 {label} 数据 (进程数: {max_workers})...")
        # initializer=init_worker 是关键：确保每个子进程启动时先登录服务器。
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = {executor.submit(sync_worker, task): task[0] for task in tasks}
            with tqdm(total=len(tasks), desc=f"同步{label}中") as pbar:
                for future in as_completed(futures):
                    try:
                        code, max_date = future.result()
                        if max_date: self.metadata[meta_type][code] = max_date
                    except Exception as e:
                        print(f"\n[Error] 任务失败: {e}")
                    pbar.update(1)
        self._save_metadata()

if __name__ == "__main__":
    manager = BaostockDataManager()
    
    print("\n--- 步骤 1: 同步核心指数数据 (基准线) ---")
    # 同步上证指数、深证成指、沪深300、创业板指、中证500、上证50
    target_indexes = ["sh.000001", "sz.399001", "sh.000300", "sz.399006", "sh.000905", "sh.000016"]
    manager.run_sync(target_indexes, is_index=True, max_workers=2)
    
    print("\n--- 步骤 2: 同步全量 A 股数据 (2006-至今) ---")
    all_stocks = manager.get_master_stock_pool(start_year=2006)
    if all_stocks:
        # 个股数据量巨大，建议使用更多进程加速
        manager.run_sync(all_stocks, is_index=False, max_workers=8)
    
    print("\n[OK] 数据同步流程全部结束。")
