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
# ✨ 核心增强 1: Numba 加速的 A 股交易约束
# =====================================================================
@njit
def fix_weights_ashare_core(ideal_w, is_limit_up, is_limit_down, is_suspended, force_liquidate):
    """极速路径依赖权重修正"""
    actual_w = np.zeros_like(ideal_w)
    for c in range(ideal_w.shape[1]):
        if not (is_suspended[0, c] or force_liquidate[0, c]):
            actual_w[0, c] = ideal_w[0, c]
            
    for t in range(1, ideal_w.shape[0]):
        for c in range(ideal_w.shape[1]):
            target = ideal_w[t, c]
            prev = actual_w[t-1, c]
            if force_liquidate[t, c]:
                actual_w[t, c] = 0.0
            elif is_suspended[t, c]:
                actual_w[t, c] = prev
            elif is_limit_up[t, c] and target > prev:
                actual_w[t, c] = prev
            elif is_limit_down[t, c] and target < prev:
                actual_w[t, c] = prev
            else:
                actual_w[t, c] = target
    return actual_w


class MultiBacktester:
    def __init__(self):
        self.engine = FactorEngine()
        
    def run_comparison(self, factors=None, top_n=20, start_date='2021-01-01', freq='5D'):
        """
        ✨ 核心增强 2: 多因子严谨对比测试 (修复 calculate_factors 缺失问题)
        """
        if factors is None:
            # 这里的名称必须匹配 FactorEngine 生成的列名
            factors = [
                ('smart_money_proxy', 'sm_20'), 
                ('amihud_illiq', 'illiq_20'), 
                ('skew', 'skew_20')
            ]
        else:
            # 假设用户传入的是 [('type', 'col_name'), ...]
            pass

        print(f"\n>>> 启动多因子对比测试 (日期: {start_date} | 频率: {freq})")
        print(f">>> 成本: 万三佣金(最低5元) + 万五滑点 + 千一印花税")
        
        # 1. 预先提取量价宽表 (统一共享)
        self.o_price = self.engine.get_pivoted_factor('open', start_date=start_date, fill_na=True)
        self.c_price = self.engine.get_pivoted_factor('close', start_date=start_date, fill_na=True)
        self.volume = self.engine.get_pivoted_factor('volume', start_date=start_date, fill_na=False).fillna(0)
        self.preclose = self.engine.get_pivoted_factor('pre_close', start_date=start_date, fill_na=True)
        
        # 2. 生成全局约束掩码
        print(">>> 正在生成全局 A 股约束掩码...")
        limit_ratios = pd.Series(0.099, index=self.o_price.columns)
        limit_ratios[self.o_price.columns.str.startswith('sz.30') | self.o_price.columns.str.startswith('sh.688')] = 0.199
        
        self.is_limit_up = (self.o_price >= (self.preclose * (1 + limit_ratios) - 0.002)).values
        self.is_limit_down = (self.o_price <= (self.preclose * (1 - limit_ratios) + 0.002)).values
        self.is_suspended = (self.volume <= 0).values
        # 识别退市/断连
        self.force_liq = (self.o_price.isna() & self.preclose.isna()).values
        
        # 获取基准
        self.benchmark_prices = self.engine.get_benchmark_prices(start_date=start_date)
        self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0.0)

        self.performance_summary = []

        # 3. 逐个提取因子并运行回测
        for f_type, f_col in factors:
            print(f"\n--- 正在处理因子: {f_col} ({f_type}) ---")
            # 通过指定 factor_name 和 windows 强制触发引擎计算
            factor_pivot = self.engine.get_pivoted_factor(
                f_col, start_date=start_date, factor_name=f_type, windows=[20], fill_na=False
            )
            
            # 正常逻辑 (趋势跟踪)
            self._backtest_one_logic(factor_pivot, f_col, start_date, top_n, ascending=False, freq=freq)
            # 反转逻辑
            self._backtest_one_logic(factor_pivot, f"{f_col}(Rev)", start_date, top_n, ascending=True, freq=freq)

        # 4. 生成排行榜
        perf_df = pd.DataFrame(self.performance_summary).sort_values('Sharpe', ascending=False)
        print("\n" + "="*130)
        print(f">>> 终极严谨版多因子绩效排行榜 (频率: {freq} | 基准: T+1 Open | 包含滑点与5元门槛费)")
        print("="*130)
        pd.options.display.float_format = '{:.4f}'.format
        print(perf_df.to_string(index=False))
        
        perf_df.to_csv('data/multi_factor_comparison.csv', index=False)
        return perf_df

    def _backtest_one_logic(self, factor_pivot, display_name, start_date, top_n, ascending, freq):
        """带有精准成本扣除的单回测模块"""
        try:
            # 提取公共日期并对齐
            common_dates = self.o_price.index.intersection(factor_pivot.index)
            c_factor = factor_pivot.reindex(index=common_dates, columns=self.o_price.columns)
            
            # 1. 计算理想权重
            # ✨ 关键修复：在排名前剔除当日未成交(volume=0)或因子值为0(通常是僵尸股)的标的
            v_mask = self.volume.loc[common_dates] > 0
            # 只有活跃股参与排名
            active_factor = c_factor.where(v_mask & (c_factor != 0), np.nan)
            
            ideal_w_raw = (active_factor.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
            step = int(freq.replace('D','')) if 'D' in freq else 1
            rebalance_days = ideal_w_raw.index[::step]
            ideal_w = ideal_w_raw.reindex(ideal_w_raw.index).loc[rebalance_days].reindex(ideal_w_raw.index).ffill().shift(1).fillna(0.0)
            
            # 2. 约束修正
            mask_idx = self.o_price.index.get_indexer(common_dates)
            actual_w_vals = fix_weights_ashare_core(
                ideal_w.values, self.is_limit_up[mask_idx], self.is_limit_down[mask_idx], 
                self.is_suspended[mask_idx], self.force_liq[mask_idx]
            )
            actual_w = pd.DataFrame(actual_w_vals, index=common_dates, columns=self.o_price.columns)
            
            # 3. VectorBT 执行
            base_comm_rate = 0.0003
            portfolio = vbt.Portfolio.from_orders(
                self.o_price.loc[common_dates], size=actual_w, size_type='targetpercent',
                init_cash=1000000, fees=base_comm_rate, slippage=0.0005,
                cash_sharing=True, group_by=True, freq='1D'
            )
            
            # 4. 精准成本后处理
            records = portfolio.orders.records_readable
            if not records.empty:
                date_col = 'Timestamp' if 'Timestamp' in records.columns else ('Index' if 'Index' in records.columns else 'Date')
                
                # 印花税
                sell_mask = records['Size'] < 0
                records['StampDuty'] = 0.0
                records.loc[sell_mask, 'StampDuty'] = records.loc[sell_mask, 'Size'].abs() * records.loc[sell_mask, 'Price'] * 0.001
                
                # 5元门槛
                records['TradeValue'] = records['Size'].abs() * records['Price']
                records['MinFeeSurcharge'] = (5.0 - records['TradeValue'] * base_comm_rate).clip(lower=0)
                
                daily_extra = records.groupby(pd.to_datetime(records[date_col]).dt.date)['StampDuty'].sum() + \
                              records.groupby(pd.to_datetime(records[date_col]).dt.date)['MinFeeSurcharge'].sum()
                
                val_series = portfolio.value()
                daily_extra.index = pd.to_datetime(daily_extra.index)
                cumulative_extra = daily_extra.reindex(val_series.index).fillna(0).cumsum()
                
                true_value_series = val_series - cumulative_extra
                strat_returns = true_value_series.pct_change().fillna(0)
            else:
                strat_returns = portfolio.returns().fillna(0.0)
            
            # 5. 指标统计
            total_ret = (1 + strat_returns).prod() - 1
            ann_ret = (1 + total_ret) ** (252 / len(strat_returns)) - 1 if len(strat_returns) > 0 else 0
            vol = strat_returns.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.03) / vol if vol != 0 else 0
            
            # 最大回撤
            cum_rets = (1 + strat_returns).cumprod()
            max_dd = ((cum_rets - cum_rets.expanding().max()) / cum_rets.expanding().max()).min()
            
            # Alpha/Beta
            mkt_returns = self.benchmark_returns.reindex(strat_returns.index).fillna(0.0)
            if len(mkt_returns) > 1:
                beta = np.cov(strat_returns, mkt_returns)[0, 1] / (np.var(mkt_returns) + 1e-9)
                alpha = (strat_returns.mean() - beta * mkt_returns.mean()) * 252
            else:
                beta, alpha = 0, 0
            
            self.performance_summary.append({
                'Factor': display_name,
                'Return[%]': total_ret * 100,
                'AnnRet[%]': ann_ret * 100,
                'Sharpe': sharpe,
                'MaxDD[%]': max_dd * 100,
                'Alpha(Ann)': alpha,
                'Beta': beta,
                'Trades': len(records)
            })
            print(f"  [DONE] {display_name:20} | Return: {total_ret*100:6.2f}% | Sharpe: {sharpe:6.4f}")
            
        except Exception as e:
            print(f"  [ERROR] {display_name} 失败: {e}")
        finally:
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 多因子横向对比排行榜')
    parser.add_argument('--freq', type=str, default='5D')
    parser.add_argument('--start', type=str, default='2021-01-01')
    args = parser.parse_args()
    
    t_start = time.time()
    MultiBacktester().run_comparison(freq=args.freq, start_date=args.start)
    print(f"\n对比任务完成！总耗时: {time.time() - t_start:.2f} 秒\n")