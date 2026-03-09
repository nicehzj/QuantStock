import vectorbt as vbt
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from factor_engine import FactorEngine
import argparse
from numba import njit
import gc

# =====================================================================
# ✨ 核心仿真引擎: Numba 加速 A 股交易约束
# =====================================================================
@njit
def fix_weights_ashare_core(ideal_w, is_limit_up, is_limit_down, is_suspended, force_liquidate):
    """
    极速路径依赖仿真逻辑。
    实现：涨停无法买入、跌停无法卖出、停牌无法交易。
    """
    actual_w = np.zeros_like(ideal_w)
    # 处理起始日
    for c in range(ideal_w.shape[1]):
        if not (is_suspended[0, c] or force_liquidate[0, c]):
            actual_w[0, c] = ideal_w[0, c]
            
    # 仿真每一天的持仓变化
    for t in range(1, ideal_w.shape[0]):
        for c in range(ideal_w.shape[1]):
            target = ideal_w[t, c]
            prev = actual_w[t-1, c]
            if force_liquidate[t, c]:
                actual_w[t, c] = 0.0 # 强制平仓
            elif is_suspended[t, c]:
                actual_w[t, c] = prev # 停牌维持原状
            elif is_limit_up[t, c] and target > prev:
                actual_w[t, c] = prev # 涨停买不进
            elif is_limit_down[t, c] and target < prev:
                actual_w[t, c] = prev # 跌停卖不出
            else:
                actual_w[t, c] = target # 正常执行
    return actual_w


class MultiBacktester:
    """
    多因子排行榜回测系统
    支持多个因子的横向对比，并严格执行 A 股交易规则与手续费模型。
    """
    def __init__(self):
        self.engine = FactorEngine()
        
    def run_comparison(self, factors=None, top_n=20, start_date='2021-01-01', freq='5D'):
        """
        运行多因子对比回测。
        factors: [(因子类型, 因子名), ...]
        """
        if factors is None:
            # 默认测试的因子列表
            factors = [
                ('smart_money_proxy', 'sm_20'), 
                ('amihud_illiq', 'illiq_20'), 
                ('skew', 'skew_20')
            ]

        print(f"\n>>> 启动对比测试 (日期: {start_date} | 频率: {freq})")
        
        # 1. 预先提取全局量价矩阵（所有因子共用，极大节省内存）
        self.o_price = self.engine.get_pivoted_factor('open', start_date=start_date, fill_na=True)
        self.c_price = self.engine.get_pivoted_factor('close', start_date=start_date, fill_na=True)
        self.volume = self.engine.get_pivoted_factor('volume', start_date=start_date, fill_na=False).fillna(0)
        self.preclose = self.engine.get_pivoted_factor('pre_close', start_date=start_date, fill_na=True)
        
        # 2. 生成 A 股全局约束布尔矩阵
        print(">>> 生成 A 股约束掩码矩阵...")
        limit_ratios = pd.Series(0.099, index=self.o_price.columns)
        limit_ratios[self.o_price.columns.str.startswith('sz.30') | self.o_price.columns.str.startswith('sh.688')] = 0.199
        
        self.is_limit_up = (self.o_price >= (self.preclose * (1 + limit_ratios) - 0.002)).values
        self.is_limit_down = (self.o_price <= (self.preclose * (1 - limit_ratios) + 0.002)).values
        self.is_suspended = (self.volume <= 0).values
        self.force_liq = (self.o_price.isna() & self.preclose.isna()).values
        
        # 获取基准收益率（如沪深300）
        self.benchmark_prices = self.engine.get_benchmark_prices(start_date=start_date)
        self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0.0)

        self.performance_summary = []

        # 3. 遍历每个因子并执行回测
        for f_type, f_col in factors:
            print(f"\n--- 处理因子: {f_col} ({f_type}) ---")
            factor_pivot = self.engine.get_pivoted_factor(f_col, start_date=start_date, factor_name=f_type, windows=[20], fill_na=False)
            
            # 同时测试 正向排名 和 反向排名 (Rev) 逻辑
            self._backtest_one_logic(factor_pivot, f_col, start_date, top_n, ascending=False, freq=freq)
            self._backtest_one_logic(factor_pivot, f"{f_col}(Rev)", start_date, top_n, ascending=True, freq=freq)

        # 4. 生成排行榜 DataFrame
        perf_df = pd.DataFrame(self.performance_summary).sort_values('Sharpe', ascending=False)
        print("\n>>> 多因子绩效对比排行榜")
        print(perf_df.to_string(index=False))
        
        perf_df.to_csv('data/multi_factor_comparison.csv', index=False)
        return perf_df

    def _backtest_one_logic(self, factor_pivot, display_name, start_date, top_n, ascending, freq):
        """核心回测模块：包含截面排名、T+1 信号处理和精准成本计算"""
        try:
            common_dates = self.o_price.index.intersection(factor_pivot.index)
            c_factor = factor_pivot.reindex(index=common_dates, columns=self.o_price.columns)
            
            # 1. 计算理想权重
            # 过滤掉当日没有成交量（停牌）的股票，不参与排名
            v_mask = self.volume.loc[common_dates] > 0
            active_factor = c_factor.where(v_mask & (c_factor != 0), np.nan)
            
            # 选排名最高的 Top N 只股票，分配 95% 总资金（留 5% 应对滑点）
            ideal_w_raw = (active_factor.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
            step = int(freq.replace('D','')) if 'D' in freq else 1
            rebalance_days = ideal_w_raw.index[::step]
            # shift(1) 模拟今天信号明天开盘成交，避免“偷看答案”
            ideal_w = ideal_w_raw.reindex(ideal_w_raw.index).loc[rebalance_days].reindex(ideal_w_raw.index).ffill().shift(1).fillna(0.0)
            
            # 2. 施加约束修正 (Numba 优化版)
            mask_idx = self.o_price.index.get_indexer(common_dates)
            actual_w_vals = fix_weights_ashare_core(
                ideal_w.values, self.is_limit_up[mask_idx], self.is_limit_down[mask_idx], 
                self.is_suspended[mask_idx], self.force_liq[mask_idx]
            )
            actual_w = pd.DataFrame(actual_w_vals, index=common_dates, columns=self.o_price.columns)
            
            # 3. VectorBT 执行模拟交易
            base_comm_rate = 0.0003 # 万三佣金
            portfolio = vbt.Portfolio.from_orders(
                self.o_price.loc[common_dates], size=actual_w, size_type='targetpercent',
                init_cash=1000000, fees=base_comm_rate, slippage=0.0005,
                cash_sharing=True, group_by=True, freq='1D'
            )
            
            # 4. 精准手续费后处理
            # VectorBT 默认手续费是按比例收取的，但在 A 股：
            # - 卖出需要额外扣除 0.1% 的印花税。
            # - 每笔交易如果手续费不足 5 元，要强行收 5 元。
            records = portfolio.orders.records_readable
            if not records.empty:
                date_col = 'Timestamp' if 'Timestamp' in records.columns else 'Date'
                
                # 计算印花税（仅在卖出记录上）
                sell_mask = records['Size'] < 0
                records['StampDuty'] = 0.0
                records.loc[sell_mask, 'StampDuty'] = records.loc[sell_mask, 'Size'].abs() * records.loc[sell_mask, 'Price'] * 0.001
                
                # 计算 5 元门槛附加费
                records['TradeValue'] = records['Size'].abs() * records['Price']
                records['MinFeeSurcharge'] = (5.0 - records['TradeValue'] * base_comm_rate).clip(lower=0)
                
                # 计算每日额外成本
                daily_extra = records.groupby(pd.to_datetime(records[date_col]).dt.date)['StampDuty'].sum() + \
                              records.groupby(pd.to_datetime(records[date_col]).dt.date)['MinFeeSurcharge'].sum()
                
                val_series = portfolio.value()
                daily_extra.index = pd.to_datetime(daily_extra.index)
                cumulative_extra = daily_extra.reindex(val_series.index).fillna(0).cumsum()
                
                # 扣除印花税和门槛费后的真实资金曲线
                true_value_series = val_series - cumulative_extra
                strat_returns = true_value_series.pct_change().fillna(0)
            else:
                strat_returns = portfolio.returns().fillna(0.0)
            
            # 5. 绩效统计指标
            total_ret = (1 + strat_returns).prod() - 1
            ann_ret = (1 + total_ret) ** (252 / len(strat_returns)) - 1 if len(strat_returns) > 0 else 0
            vol = strat_returns.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.03) / vol if vol != 0 else 0
            
            self.performance_summary.append({
                'Factor': display_name,
                'Return[%]': total_ret * 100,
                'AnnRet[%]': ann_ret * 100,
                'Sharpe': sharpe,
                'Trades': len(records)
            })
            print(f"  [DONE] {display_name:20} | Return: {total_ret*100:6.2f}% | Sharpe: {sharpe:6.4f}")
            
        except Exception as e:
            print(f"  [ERROR] {display_name} 失败: {e}")
        finally:
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多因子横向对比排行榜')
    parser.add_argument('--freq', type=str, default='5D')
    parser.add_argument('--start', type=str, default='2021-01-01')
    args = parser.parse_args()
    
    t_start = time.time()
    MultiBacktester().run_comparison(freq=args.freq, start_date=args.start)
    print(f"\n任务完成！总耗时: {time.time() - t_start:.2f} 秒\n")
