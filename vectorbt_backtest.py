import vectorbt as vbt
import pandas as pd
import os
import time
from factor_engine import FactorEngine

class VBTBacktester:
    """
    向量化回测引擎 (Vectorbt)
    利用 NumPy/Numba 的加速能力，对全市场数千只标的进行瞬间回测。
    主要用于：因子初步筛选、大样本参数优化。
    """
    
    def __init__(self):
        # 复用之前的因子引擎获取数据路径
        self.engine = FactorEngine()
        
    def run_momentum_screening(self, top_n=20, fees=0.0013, start_date='2021-01-01', end_date=None):
        """
        全市场动量选股策略快速回测。
        策略逻辑：每日根据 20 日动量因子排序，持有最强的 top_n 只股票，次日全额调仓。
        :param top_n: 每日持仓股票数量
        :param fees: 综合摩擦成本（佣金+滑点估算）
        """
        start_time = time.time()
        
        # 1. 加载因子数据
        factors_df = self.engine.calculate_basic_factors()
        
        # 2. 准备价格矩阵
        # Vectorbt 需要宽表格式：Index 为时间，Columns 为股票代码，Values 为收盘价
        print(">>> 正在提取价格矩阵 (DuckDB 极速扫描)...")
        sql = f"SELECT date, code, close FROM read_parquet('{self.engine.parquet_glob}')"
        price_df = self.engine.db.execute(sql).df().pivot(index='date', columns='code', values='close')
        price_df.index = pd.to_datetime(price_df.index)
        
        # 3. 日期过滤：切分样本外区间 (Out-of-Sample)
        if end_date is None:
            end_date = price_df.index.max()
        
        price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        factors_df = factors_df[(factors_df['date'] >= start_date) & (factors_df['date'] <= end_date)]

        if price_df.empty or factors_df.empty:
            print(f"[Warning] 错误: 在区间 {start_date} -> {end_date} 内没有足够的回测数据。")
            return None

        # 处理缺失值（退市或停牌股票），保持矩阵完整性
        price_df = price_df.ffill()

        # 4. 生成交易信号 (Entries)
        print(f">>> 正在根据动量因子生成 Top {top_n} 信号矩阵...")
        # 将因子 DataFrame 转换为与价格矩阵对齐的宽表
        factor_pivot = factors_df.pivot(index='date', columns='code', values='factor_mom_20')
        factor_pivot.index = pd.to_datetime(factor_pivot.index)
        
        # 获取前收盘价用于计算涨跌停限制
        sql_pre = f"SELECT date, code, preclose FROM read_parquet('{self.engine.parquet_glob}')"
        preclose_df = self.engine.db.execute(sql_pre).df().pivot(index='date', columns='code', values='preclose')
        preclose_df.index = pd.to_datetime(preclose_df.index)

        # --- 核心修复：强制对齐所有矩阵 ---
        common_index = price_df.index.intersection(factor_pivot.index).intersection(preclose_df.index)
        common_columns = price_df.columns.intersection(factor_pivot.columns).intersection(preclose_df.columns)
        
        price_df = price_df.loc[common_index, common_columns]
        factor_pivot = factor_pivot.loc[common_index, common_columns]
        preclose_df = preclose_df.loc[common_index, common_columns]
        
        # A 股限制逻辑 1：计算动态涨跌停 Mask
        limit_ratios = pd.Series(0.099, index=common_columns)
        limit_ratios[common_columns.str.startswith('sz.30') | common_columns.str.startswith('sh.688')] = 0.199
        
        is_limit_up = price_df >= preclose_df * (1 + limit_ratios)
        is_limit_down = price_df <= preclose_df * (1 - limit_ratios)

        # 对每日截面进行排名，筛选前 top_n
        is_in_top_n = factor_pivot.rank(axis=1, ascending=True, method='first') <= top_n
        
        # 生成精准的进入/退出意向 (这是信号日逻辑)
        raw_entries_sig = is_in_top_n & (~is_in_top_n.shift(1).fillna(False))
        raw_exits_sig = (~is_in_top_n) & (is_in_top_n.shift(1).fillna(False))

        # 模拟 T+1 转换到执行日
        exec_entries = raw_entries_sig.shift(1).fillna(False)
        exec_exits = raw_exits_sig.shift(1).fillna(False)

        # 在执行日应用涨跌停限制
        entries = exec_entries & (~is_limit_up)
        exits = exec_exits & (~is_limit_down)

        entries = entries.astype(bool)
        exits = exits.astype(bool)
        price_df = price_df.ffill().bfill().astype(float)
        
        # 4. 执行向量化回测
        print(f">>> Vectorbt 引擎正在并行计算收益率曲线 (标的数: {len(common_columns)})...")
        
        portfolio = vbt.Portfolio.from_signals(
            price_df, 
            entries=entries, 
            exits=exits, 
            size=0.95 / top_n, 
            size_type='percent',
            min_size=100,             
            size_granularity=100,     
            freq='1D',
            init_cash=1000000,        
            fees=fees,
            cash_sharing=True,
            call_seq='auto',
            group_by=True
        )

        duration = time.time() - start_time
        print(f"[OK] 全市场回测完成！总耗时: {duration:.2f} 秒")

        # --- 核心新增：生成统一交易流水日志 ---
        print(">>> 正在生成全市场统一交易流水日志 (Unified Trade Log)...")
        os.makedirs('data', exist_ok=True)
        
        try:
            trades_df = pd.DataFrame(portfolio.trades.values)
        except Exception:
            trades_df = portfolio.trades.records_df
        
        import numpy as np
        entry_dates, entry_cols = np.where(exec_entries.values) 
        exit_dates, exit_cols = np.where(exec_exits.values) 
        
        unified_records = []

        # 处理买入意向
        for d_idx, c_idx in zip(entry_dates, entry_cols):
            date = common_index[d_idx]
            code = common_columns[c_idx]
            price = price_df.iloc[d_idx, c_idx]
            status = "Success"
            reason = ""
            size = 0
            
            if is_limit_up.iloc[d_idx, c_idx]:
                status = "Failed"
                reason = "Limit Up (Blocked)"
            else:
                match = trades_df[(trades_df['col'] == c_idx) & (trades_df['entry_idx'] == d_idx)]
                if not match.empty:
                    size = match.iloc[0]['size']
                else:
                    status = "Failed"
                    reason = "Insufficient Funds / Lot Size Limit"
            
            unified_records.append({
                'Timestamp': date, 'Code': code, 'Action': 'BUY',
                'Price': round(price, 3), 'Status': status, 'Reason': reason,
                'Size': size, 'PnL': 0, 'PnL_Pct': 0
            })

        # 处理卖出意向
        for d_idx, c_idx in zip(exit_dates, exit_cols):
            date = common_index[d_idx]
            code = common_columns[c_idx]
            price = price_df.iloc[d_idx, c_idx]
            status = "Success"
            reason = ""
            size = 0
            pnl = 0
            pnl_pct = 0
            
            if is_limit_down.iloc[d_idx, c_idx]:
                status = "Failed"
                reason = "Limit Down (Blocked)"
            else:
                match = trades_df[(trades_df['col'] == c_idx) & (trades_df['exit_idx'] == d_idx)]
                if not match.empty:
                    size = match.iloc[0]['size']
                    pnl = match.iloc[0]['pnl']
                    pnl_pct = match.iloc[0]['return'] * 100 
                else:
                    status = "Ignored"
                    reason = "No Position to Sell"
            
            unified_records.append({
                'Timestamp': date, 'Code': code, 'Action': 'SELL',
                'Price': round(price, 3), 'Status': status, 'Reason': reason,
                'Size': size, 'PnL': round(pnl, 2), 'PnL_Pct': round(pnl_pct, 2)
            })

        unified_df = pd.DataFrame(unified_records)
        if not unified_df.empty:
            unified_df = unified_df.sort_values(['Timestamp', 'Code']).reset_index(drop=True)
            unified_df.to_csv('data/vbt_unified_trade_log.csv', index=False)
            print(f"[OK] 统一交易日志已生成: data/vbt_unified_trade_log.csv (共 {len(unified_df)} 条记录)")
        else:
            print("[Warning] 未产生任何交易信号或记录。")

        stats = portfolio.stats()
        print("\n" + "="*50)
        print("Vectorbt 全市场动量策略统计报告")
        print("-" * 50)
        def get_stat(keys):
            for k in keys:
                if k in stats: return stats[k]
            return 0.0

        ann_ret = get_stat(['Ann. Return [%]', 'Annualized Return [%]', 'Total Return [%]'])
        max_dd = get_stat(['Max. Drawdown [%]', 'Max Drawdown [%]'])
        sharpe = get_stat(['Sharpe Ratio'])
        trades = get_stat(['Total Trades'])

        print(f"策略表现: {ann_ret:.2f}%")
        print(f"最大回撤: {max_dd:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"总交易次数: {int(trades)}")
        print("="*50)
        
        return portfolio

if __name__ == "__main__":
    try:
        tester = VBTBacktester()
        portfolio = tester.run_momentum_screening(top_n=20)
    except Exception as e:
        print(f"[Error] 回测运行失败: {e}")
