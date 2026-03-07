import pandas as pd
import numpy as np
import vectorbt as vbt
import os
import matplotlib.pyplot as plt
import seaborn as sns
from factor_engine import FactorEngine
from tqdm import tqdm
import gc
import argparse
import time

class VectorizedOptimizer:
    def __init__(self):
        self.engine = FactorEngine()
        
    def get_all_factor_variants_fast(self, factor_name, windows, price_fields):
        # 1. 计算所有变体长表
        big_df_sample = self.engine.calculate_all_variants(factor_name, windows)
        
        # 2. 针对每个 window，获取因子宽表
        variants = {}
        for w in windows:
            base_col = [c for c in big_df_sample.columns if f"_{w}" in c][0]
            pivot = self.engine.get_pivoted_factor(base_col, factor_name=factor_name, windows=windows, fill_na=False)
            pivot.index = pd.to_datetime(pivot.index)
            # MAD 去极值
            m = pivot.median(axis=1)
            mad = (pivot.sub(m, axis=0)).abs().median(axis=1)
            pivot = pivot.clip(m - 4.4478*mad, m + 4.4478*mad, axis=0)
            variants[w] = pivot.astype(np.float32)
            
        # 3. 预取价格宽表并执行【数值净化】
        prices = {}
        for pf in price_fields:
            df_p = self.engine.get_pivoted_factor(pf, fill_na=True).ffill().bfill()
            # 关键修复：确保所有执行价格 > 0 且有限，防止 Numba 崩溃
            df_p = df_p.clip(lower=0.001).fillna(0.001)
            df_p.index = pd.to_datetime(df_p.index)
            prices[pf] = df_p.astype(np.float32)
            
        volume = self.engine.get_pivoted_factor('volume', fill_na=False).fillna(0).astype(np.float32)
        volume.index = pd.to_datetime(volume.index)
        
        pre_close = self.engine.get_pivoted_factor('pre_close', fill_na=True).ffill().bfill()
        pre_close = pre_close.clip(lower=0.001).fillna(0.001) # 同步净化
        pre_close.index = pd.to_datetime(pre_close.index)
            
        return variants, prices, volume, pre_close.astype(np.float32)

    def run_fast_optimize(self, factor_name, mode='smoke', batch_size=5):
        if mode == 'smoke':
            print(f"\n>>> 启动【快速冒烟测试】网格搜索: {factor_name}")
            windows = [10, 20]
            price_fields = ['close']
            freqs = ['5D']
        else:
            print(f"\n>>> 启动【全维度向量化】高性能网格搜索: {factor_name}")
            windows = [5, 10, 20, 40]
            price_fields = ['close', 'vwap']
            freqs = ['1D', '5D', '10D', '20D']

        variants, prices, volume, pre_close = self.get_all_factor_variants_fast(factor_name, windows, price_fields)
        
        # 预计算流动性掩码
        is_tradable = (volume > 0).astype(bool)
        price_base = prices['close'] 
        ret_daily = (price_base / pre_close - 1).fillna(0)
        can_trade = is_tradable & (ret_daily < 0.098) & (ret_daily > -0.098)
        
        top_n = 20
        signal_bank = {}
        print(">>> 正在预计算截面排名信号...")
        for w, f_pivot in variants.items():
            for d in [True, False]: 
                rank_mat = f_pivot.rank(axis=1, ascending=d, method='first')
                signal_bank[(w, d)] = (rank_mat <= top_n).astype(np.float32) * (0.95/top_n)
        
        all_params = []
        for w in windows:
            for pf in price_fields:
                for freq in freqs:
                    for d in [True, False]:
                        all_params.append({'w':w, 'p':pf, 'f':freq, 'd':d})

        all_results = []
        num_batches = (len(all_params) + batch_size - 1) // batch_size
        
        for b in range(num_batches):
            batch_params = all_params[b*batch_size : (b+1)*batch_size]
            print(f">>> 批次 {b+1}/{num_batches} (执行中...)")
            
            batch_weights = []; keys = []
            for p in batch_params:
                base_sig = signal_bank[(p['w'], p['d'])]
                step = int(p['f'].replace('D',''))
                rebalance_days = base_sig.index[::step]
                final_w = base_sig.reindex(base_sig.index).loc[rebalance_days].reindex(base_sig.index).ffill().shift(1).fillna(0.0)
                batch_weights.append(final_w.astype(np.float32))
                keys.append((p['w'], p['p'], p['f'], p['d']))
            
            mega_w = pd.concat(batch_weights, axis=1, keys=keys, names=['w', 'p', 'f', 'rev'])
            # 流动性锁死：不可交易日权重设为 NaN 跳过订单
            mega_w = mega_w.where(can_trade, np.nan)
            
            o_p = prices[batch_params[0]['p']]
            portfolio = vbt.Portfolio.from_orders(
                o_p, size=mega_w, size_type='targetpercent', 
                init_cash=1000000, fees=0.0015, freq='1D', 
                group_by=['w', 'p', 'f', 'rev'], cash_sharing=True
            )
            
            batch_stats = pd.DataFrame({
                'Return': portfolio.total_return() * 100,
                'Sharpe': portfolio.sharpe_ratio(),
                'Calmar': portfolio.calmar_ratio(),
                'MaxDD': portfolio.max_drawdown() * 100
            }).reset_index()
            all_results.append(batch_stats)
            del portfolio, mega_w; gc.collect()

        stats_df = pd.concat(all_results).reset_index(drop=True)
        stats_df.columns = ['w', 'p', 'f', 'rev', 'Return', 'Sharpe', 'Calmar', 'MaxDD']
        
        best_row = stats_df.loc[stats_df['Sharpe'].idxmax()]
        print(f"\n[OK] 锁定最优配置: W:{best_row['w']} P:{best_row['p']} F:{best_row['f']} D:{best_row['rev']} Sharpe:{best_row['Sharpe']:.4f}")

        # OOS 盲测
        print(f"\n>>> 启动【2024-2026】盲测考场...")
        test_period = '2024-01-01'
        final_res = self.rigorous_backtest_single(
            variants[best_row['w']].loc[test_period:], 
            prices[best_row['p']].loc[test_period:], 
            volume.loc[test_period:], 
            pre_close.loc[test_period:], 
            freq=best_row['f'], ascending=best_row['rev']
        )
        print("\n" + "*"*60 + "\n因子终极盲测报告 (2024-2026)\n" + "*"*60)
        oos_stats = final_res.stats()
        print(oos_stats)
        
        # 导出 OOS 报告
        oos_stats.to_csv(f'data/oos_report_{factor_name}.csv')
        with open(f'data/oos_report_{factor_name}.md', 'w', encoding='utf-8') as f:
            f.write(f"# 因子盲测体检报告: {factor_name}\n\n")
            f.write(f"## 最优参数配置\n- 计算窗口 (W): {best_row['w']}\n- 价格基准 (P): {best_row['p']}\n- 调仓频率 (F): {best_row['f']}\n- 方向 (D): {best_row['rev']}\n\n")
            f.write("## 核心回测指标 (2024-2026)\n\n")
            f.write("| 指标 | 数值 |\n| :--- | :--- |\n")
            for idx, val in oos_stats.items():
                f.write(f"| {idx} | {val} |\n")
        print(f"\n[OK] 盲测详细指标已导出至: data/oos_report_{factor_name}.md/.csv")
        
        plt.figure(figsize=(12,6))
        final_res.value().plot(label=f'Strategy (W:{best_row["w"]})', color='red', lw=2)
        benchmark_p = self.engine.get_benchmark_prices(start_date=test_period)
        if not benchmark_p.empty:
            benchmark_normalized = (benchmark_p / benchmark_p.iloc[0]) * final_res.value().iloc[0]
            benchmark_normalized.plot(label='Benchmark (HS300)', color='black', alpha=0.5, linestyle='--')
        plt.title(f'Final OOS Test with Liquidity Lock: {factor_name}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f'data/final_oos_{factor_name}.png')

        self.run_historical_rolling_test(factor_name, variants[best_row['w']], prices[best_row['p']], volume, pre_close, best_row)
        return stats_df

    def run_historical_rolling_test(self, factor_name, f_pivot, o_p, volume, pre_close, best_params):
        print(f"\n" + "="*60 + "\n正在进入 2006-2023 历史压力测试 (集成流动性过滤)...\n" + "="*60)
        windows = []
        for y in range(2006, 2022):
            windows.append((f"{y}-01-01", f"{y+2}-12-31"))
            
        rolling_results = []
        for start, end in tqdm(windows, desc="滚动窗口扫描"):
            try:
                slice_f = f_pivot.loc[start:end]
                slice_p = o_p.loc[start:end]
                slice_v = volume.loc[start:end]
                slice_pre = pre_close.loc[start:end]
                if len(slice_f) < 100: continue
                
                res = self.rigorous_backtest_single(slice_f, slice_p, slice_v, slice_pre, freq=best_params['f'], ascending=best_params['rev'])
                s = res.stats()
                trades_rec = res.trades.records_readable
                avg_trade_ret = trades_rec['Return'].mean() * 100 if not trades_rec.empty else 0
                
                years = len(slice_f) / 252
                order_records = res.orders.records_readable
                total_trade_volume = (order_records['Size'].abs() * order_records['Price']).sum() if not order_records.empty else 0
                avg_portfolio_value = res.value().mean().mean()
                ann_turnover = (total_trade_volume / 2) / avg_portfolio_value / years if (years > 0 and avg_portfolio_value > 0) else 0
                
                rolling_results.append({
                    'Window': f"{start[:4]}-{end[:4]}",
                    'Return': s.get('Total Return [%]', 0),
                    'Sharpe': s.get('Sharpe Ratio', 0),
                    'Calmar': s.get('Calmar Ratio', 0),
                    'MaxDD': s.get('Max Drawdown [%]', 0),
                    'WinRate': s.get('Win Rate [%]', 0),
                    'ProfitFactor': s.get('Profit Factor', 0),
                    'Turnover': ann_turnover,
                    'AvgTradeRet': avg_trade_ret
                })
            except Exception: continue

        rolling_df = pd.DataFrame(rolling_results)
        if rolling_df.empty: return
        print("\n" + "*"*60 + "\n历史滚动“体检”核心指标汇总 (2006-2023)\n" + "*"*60)
        print(rolling_df.to_string(index=False))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35)
        axes[0,0].bar(rolling_df['Window'], rolling_df['Sharpe'], color='skyblue', label='Sharpe')
        axes[0,0].plot(rolling_df['Window'], rolling_df['Calmar'], marker='o', color='gold', label='Calmar')
        axes[0,0].set_title('Consistency: Sharpe & Calmar'); axes[0,0].legend(); axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,1].plot(rolling_df['Window'], rolling_df['WinRate'], marker='o', color='forestgreen'); axes[0,1].set_title('Win Rate Trend'); axes[0,1].tick_params(axis='x', rotation=45)
        axes[1,0].bar(rolling_df['Window'], rolling_df['MaxDD'], color='salmon'); axes[1,0].set_title('Max Drawdown (%)'); axes[1,0].invert_yaxis(); axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,1].bar(rolling_df['Window'], rolling_df['Turnover'], color='gray', alpha=0.6); axes[1,1].set_title('Annualized Turnover'); axes[1,1].tick_params(axis='x', rotation=45)
        plt.savefig(f'data/rolling_diagnostic_{factor_name}.png')
        rolling_df.to_csv(f'data/rolling_diagnostic_{factor_name}.csv', index=False, encoding='utf-8-sig')

    def rigorous_backtest_single(self, f_pivot, o_p, volume, pre_close, top_n=20, freq='5D', ascending=True):
        ideal_w = (f_pivot.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
        step = int(freq.replace('D',''))
        rebalance_days = ideal_w.index[::step]
        final_w = ideal_w.reindex(ideal_w.index).loc[rebalance_days].reindex(ideal_w.index).ffill().shift(1).fillna(0.0)
        
        is_tradable = (volume > 0).astype(bool)
        ret_daily = (o_p / pre_close - 1).fillna(0)
        can_trade = is_tradable & (ret_daily < 0.098) & (ret_daily > -0.098)
        
        final_w = final_w.where(can_trade, np.nan)
        # 数值净化：确保 VectorBT 获得的始终是正值价格矩阵
        valid_p = o_p.ffill().bfill().clip(lower=0.001).fillna(0.001)
        
        return vbt.Portfolio.from_orders(valid_p, size=final_w, size_type='targetpercent', init_cash=1000000, fees=0.0015, freq='1D', cash_sharing=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 因子高性能优化器')
    parser.add_argument('--factor', type=str, default='smart_money_proxy', help='因子名称')
    parser.add_argument('--mode', type=str, choices=['smoke', 'full'], default='smoke', help='运行模式: smoke(冒烟) 或 full(全量)')
    args = parser.parse_args()
    t_start = time.time()
    VectorizedOptimizer().run_fast_optimize(args.factor, mode=args.mode)
    print(f"\n============================================================\n任务完成！总耗时: {time.time() - t_start:.2f} 秒\n============================================================\n")
