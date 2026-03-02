import baostock as bs
import pandas as pd
import os
import json
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta

# ==========================================
# 全局变量：用于子进程保持登录 Session
# ==========================================
_child_initialized = False

def init_worker():
    """每个子进程启动时执行一次登录"""
    global _child_initialized
    if not _child_initialized:
        bs.login()
        _child_initialized = True

def sync_worker(task_info):
    """
    单个股票/指数的同步任务
    """
    code, is_index, start_date, end_date, save_dir = task_info
    if start_date > end_date:
        return code, None

    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg" if is_index else \
             "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    adjustflag = "3" if is_index else "1"

    data_rows = []
    success = False
    
    try:
        # 子进程已经通过 init_worker 登录
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
                time.sleep(random.random() * 2)
    except:
        pass

    if not success or not data_rows:
        # 如果是老股票退市了，标记为已同步到最新，避免下次重复拉取
        return code, end_date

    new_df = pd.DataFrame(data_rows, columns=rs.fields)
    numeric_cols = [c for c in new_df.columns if c not in ['date', 'code', 'tradestatus', 'isST']]
    new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    file_path = os.path.join(save_dir, f"{code}.parquet")
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date']).sort_values('date')
        combined_df.to_parquet(file_path, index=False, compression='snappy')
        max_date = combined_df['date'].max()
    else:
        new_df.to_parquet(file_path, index=False, compression='snappy')
        max_date = new_df['date'].max()
    
    return code, max_date

class BaostockDataManager:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.stock_path = os.path.join(base_path, "daily_k")
        self.index_path = os.path.join(base_path, "index_k")
        self.pool_cache_path = os.path.join(base_path, "stock_pools")
        self.metadata_file = os.path.join(base_path, "sync_metadata.json")
        
        os.makedirs(self.stock_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.pool_cache_path, exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"stocks": {}, "indexes": {}}

    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def _get_effective_date(self):
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
        if last_trading_day == today_str and now.hour < 20:
            return trading_days[-2] if len(trading_days) >= 2 else last_trading_day
        return last_trading_day

    def get_master_stock_pool(self, start_year=2006):
        """
        核心优化：获取从 2006 年至今每年初在市股票的并集，彻底解决幸存者偏差。
        增加缓存逻辑：避免重复下载。
        """
        master_pool_file = os.path.join(self.pool_cache_path, "master_pool.json")
        
        # 1. 尝试直接加载最终的总池 (如果已经存在)
        if os.path.exists(master_pool_file):
            with open(master_pool_file, 'r', encoding='utf-8') as f:
                cached_pool = json.load(f)
                print(f"✅ 从本地加载全局股票池，共计 {len(cached_pool)} 只股票。")
                return cached_pool

        print(f">>> 正在构建从 {start_year} 年至今的全局股票池...")
        current_year = datetime.now().year
        master_pool = set()
        
        bs.login()
        for year in range(start_year, current_year + 1):
            year_cache_file = os.path.join(self.pool_cache_path, f"stock_pool_{year}.json")
            
            # 2. 检查该年份是否已有本地缓存
            if os.path.exists(year_cache_file):
                with open(year_cache_file, 'r', encoding='utf-8') as f:
                    year_stocks = json.load(f)
                    master_pool.update(year_stocks)
                    print(f"  - {year} 年数据: 已从本地加载 ({len(year_stocks)} 只)")
                    continue

            # 3. 本地无缓存，从 Baostock 获取
            target_date = f"{year}-01-01"
            rs_date = bs.query_trade_dates(start_date=target_date, end_date=f"{year}-01-20")
            df_date = rs_date.get_data()
            if not df_date.empty:
                valid_days = df_date[df_date['is_trading_day'] == '1']
                if not valid_days.empty:
                    first_day = valid_days.iloc[0]['calendar_date']
                    rs = bs.query_all_stock(day=first_day)
                    df = rs.get_data()
                    if not df.empty:
                        # 过滤逻辑：sz.0-3 开头，sh.6 开头
                        stocks = df[df['code'].str.contains(r'^sz\.[0-3]|^sh\.6')]['code'].tolist()
                        master_pool.update(stocks)
                        
                        # 保存该年份的缓存
                        with open(year_cache_file, 'w', encoding='utf-8') as f:
                            json.dump(stocks, f, indent=4)
                        print(f"  - {year} 年数据: 已从 Baostock 获取并缓存 ({len(stocks)} 只)")
        
        bs.logout()
        sorted_pool = sorted(list(master_pool))
        
        # 4. 保存最终的合并总池
        with open(master_pool_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_pool, f, indent=4)
            
        print(f"✅ 全局股票池构建完成，共计 {len(sorted_pool)} 只股票 (包含已退市)。")
        return sorted_pool

    def run_sync(self, codes, is_index=False, max_workers=4):
        label = "指数" if is_index else "个股"
        meta_type = "indexes" if is_index else "stocks"
        save_dir = self.index_path if is_index else self.stock_path
        end_date = self._get_effective_date()
        
        tasks = []
        for code in codes:
            last = self.metadata[meta_type].get(code, "2006-01-01")
            start = last if last == "2006-01-01" else (pd.to_datetime(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((code, is_index, start, end_date, save_dir))

        print(f">>> 启动多进程同步 {label} 数据 (进程数: {max_workers})...")
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = {executor.submit(sync_worker, task): task[0] for task in tasks}
            with tqdm(total=len(tasks), desc=f"同步{label}中") as pbar:
                for future in as_completed(futures):
                    try:
                        code, max_date = future.result()
                        if max_date: self.metadata[meta_type][code] = max_date
                    except Exception as e:
                        print(f"\n❌ 任务失败: {e}")
                    pbar.update(1)
        self._save_metadata()

if __name__ == "__main__":
    manager = BaostockDataManager()
    
    # 1. 指数数据 (固定几只核心指数)
    print("\n--- 步骤 1: 同步核心指数数据 ---")
    target_indexes = ["sh.000001", "sz.399001", "sh.000300", "sz.399006", "sh.000905", "sh.000016"]
    manager.run_sync(target_indexes, is_index=True, max_workers=2)
    
    # 2. 全量股票池同步 (自动规避幸存者偏差)
    print("\n--- 步骤 2: 同步全局股票池 (2006-至今) ---")
    # 为了测试，我们先取全局池的前 40 只进行同步，你可以改为 all_stocks 全量运行
    all_stocks = manager.get_master_stock_pool(start_year=2006)
    if all_stocks:
        manager.run_sync(all_stocks, is_index=False, max_workers=8)
    
    print("\n✅ 所有任务已结束。")
