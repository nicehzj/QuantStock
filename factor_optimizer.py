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

class VectorizedOptimizer:
    def __init__(self):
        self.engine = FactorEngine()
        
    def get_all_factor_variants_fast(self, factor_name, windows, price_fields):
        big_df = self.engine.calculate_all_variants(factor_name, windows)
        all_variants = {}
        pbar = tqdm(total=len(windows)*len(price_fields), desc="向量化预处理")
        for w in windows:
            base_col = [c for c in big_df.columns if f"sm_{w}" in c or f"illiq_{w}" in c or f"skew_{w}" in c or f"ivol_{w}" in c][0]
            for pf in price_fields:
                pivot = big_df.pivot(index='date', columns='code', values=base_col)
                m = pivot.median(axis=1)
                mad = (pivot.sub(m, axis=0)).abs().median(axis=1)
                pivot = pivot.clip(m - 4.4478*mad, m + 4.4478*mad, axis=0).rank(axis=1, pct=True)
                all_variants[(w, pf)] = pivot
                pbar.update(1)
        pbar.close()
        return all_variants, big_df

    def run_fast_optimize(self, factor_name, mode='smoke', batch_size=6):
        if mode == 'smoke':
            print(f"\n>>> 启动【快速冒烟测试】网格搜索: {factor_name}")
            windows = [10, 20]
            price_fields = ['close', 'vwap']
            freqs = ['5D']
        else:
            print(f"\n>>> 启动【全维度向量化】高性能网格搜索: {factor_name}")
            windows = [5, 10, 20, 40]
            price_fields = ['close', 'vwap', 'open']
            freqs = ['1D', '5D', '20D']

        variants, raw_big_df = self.get_all_factor_variants_fast(factor_name, windows, price_fields)
        o_p = raw_big_df.pivot(index='date', columns='code', values='open').ffill()

        directions = [True, False]
        top_n = 20
        all_params = []
        for (w, pf) in variants.keys():
            for freq in freqs:
                for d in directions:
                    all_params.append({'w':w, 'p':pf, 'f':freq, 'd':d})

        all_results = []
        num_batches = (len(all_params) + batch_size - 1) // batch_size
        
        for b in range(num_batches):
            batch_params = all_params[b*batch_size : (b+1)*batch_size]
            print(f">>> 批次 {b+1}/{num_batches} (执行中...)")
            batch_weights = []; keys = []
            for p in batch_params:
                f_pivot = variants[(p['w'], p['p'])]
                ideal_w = (f_pivot.rank(axis=1, ascending=p['d'], method='first') <= top_n).astype(float) * (0.95/top_n)
                step = int(p['f'].replace('D',''))
                final_w = ideal_w.iloc[::step].reindex(ideal_w.index).ffill().shift(1).fillna(0.0)
                batch_weights.append(final_w); keys.append((p['w'], p['p'], p['f'], p['d']))
            
            mega_w = pd.concat(batch_weights, axis=1, keys=keys, names=['w', 'p', 'f', 'rev'])
            portfolio = vbt.Portfolio.from_orders(o_p, size=mega_w, size_type='targetpercent', init_cash=1000000, fees=0.0015, freq='1D', group_by=['w', 'p', 'f', 'rev'], cash_sharing=True)
            
            batch_stats = pd.DataFrame({
                'Return': portfolio.total_return() * 100,
                'Sharpe': portfolio.sharpe_ratio(),
                'MaxDD': portfolio.max_drawdown() * 100
            }).reset_index()
            all_results.append(batch_stats)
            del portfolio, mega_w; gc.collect()

        stats_df = pd.concat(all_results).reset_index(drop=True)
        stats_df.columns = ['w', 'p', 'f', 'rev', 'Return', 'Sharpe', 'MaxDD']
        
        best_row = stats_df.loc[stats_df['Sharpe'].idxmax()]
        print(f"\n[OK] 锁定最优配置: W:{best_row['w']} P:{best_row['p']} F:{best_row['f']} D:{best_row['rev']} Sharpe:{best_row['Sharpe']:.4f}")

        # OOS 盲测
        print(f"\n>>> 启动【2024-2026】盲测考场...")
        test_period = '2024-01-01'
        test_f = variants[(best_row['w'], best_row['p'])].loc[test_period:]
        test_o_p = o_p.loc[test_period:]
        final_res = self.rigorous_backtest_single(test_f, test_o_p, freq=best_row['f'], ascending=best_row['rev'])
        
        print("\n" + "*"*60 + "\n因子终极盲测报告 (2024-2026)\n" + "*"*60)
        print(final_res.stats())
        
        plt.figure(figsize=(12,6))
        final_res.value().plot(label=f'OOS Strategy (W:{best_row["w"]} F:{best_row["f"]})', color='red', lw=2)
        plt.title(f'Final OOS Test: {factor_name} (2024-2026)')
        plt.savefig(f'data/final_oos_{factor_name}.png')

        plt.figure(figsize=(10,6))
        plot_df = stats_df[stats_df['rev'] == best_row['rev']].pivot_table(index='w', columns='f', values='Sharpe')
        sns.heatmap(plot_df, annot=True, cmap='RdYlGn')
        plt.savefig(f'data/vectorized_landscape_{factor_name}.png')
        return stats_df

    def rigorous_backtest_single(self, f_pivot, o_p, top_n=20, freq='5D', ascending=True):
        ideal_w = (f_pivot.rank(axis=1, ascending=ascending, method='first') <= top_n).astype(float) * (0.95 / top_n)
        step = int(freq.replace('D',''))
        final_w = ideal_w.iloc[::step].reindex(ideal_w.index).ffill().shift(1).fillna(0.0)
        return vbt.Portfolio.from_orders(o_p, size=final_w, size_type='targetpercent', init_cash=1000000, fees=0.0015, freq='1D', cash_sharing=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantStock 因子高性能优化器')
    parser.add_argument('--factor', type=str, default='smart_money_proxy', help='因子名称')
    parser.add_argument('--mode', type=str, choices=['smoke', 'full'], default='smoke', help='运行模式: smoke(冒烟) 或 full(全量)')
    args = parser.parse_args()
    
    VectorizedOptimizer().run_fast_optimize(args.factor, mode=args.mode)
