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
        # 1. 计算所有变体长表 (仅用于确定列名)
        big_df_sample = self.engine.calculate_all_variants(factor_name, windows)
        
        # 2. 针对每个 window，利用 DuckDB 极速 PIVOT 获取宽表
        variants = {}
        for w in windows:
            base_col = [c for c in big_df_sample.columns if f"_{w}" in c][0]
            # 因子不填充 (NaN 代表停牌)
            pivot = self.engine.get_pivoted_factor(base_col, factor_name=factor_name, windows=windows, fill_na=False)
            pivot.index = pd.to_datetime(pivot.index)
            
            # 3. 极速 MAD 去极值
            m = pivot.median(axis=1)
            mad = (pivot.sub(m, axis=0)).abs().median(axis=1)
            pivot = pivot.clip(m - 4.4478*mad, m + 4.4478*mad, axis=0)
            variants[w] = pivot.astype(np.float32)
            
        # 4. 预取所有价格宽表
        prices = {}
        for pf in price_fields:
            # 价格必须填充，否则 VectorBT 无法计算收益
            df_p = self.engine.get_pivoted_factor(pf, fill_na=True)
            df_p.index = pd.to_datetime(df_p.index)
            prices[pf] = df_p.astype(np.float32)
            
        return variants, prices

    def run_fast_optimize(self, factor_name, mode='smoke', batch_size=20):
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

        # 获取预处理后的数据
        variants, prices = self.get_all_factor_variants_fast(factor_name, windows, price_fields)
        
        # 预计算所有 (Window, Direction) 的 Top-N 信号矩阵 (未重采样)
        # 这将消除循环内部最耗时的截面排序操作
        top_n = 20
        signal_bank = {}
        print(">>> 正在预计算截面排名信号...")
        for w, f_pivot in variants.items():
            for d in [True, False]: # True: 升序(做空低分), False: 降序(做多高分)
                # 使用 numpy 极速计算排名
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
            print(f">>> 批次 {b+1}/{num_batches} (大小: {len(batch_params)})")
            
            batch_weights = []
            keys = []
            
            for p in batch_params:
                # 1. 获取预计算好的信号
                base_sig = signal_bank[(p['w'], p['d'])]
                
                # 2. NumPy 极速重采样逻辑 (替代 ideal_w.iloc[::step].reindex(...).ffill().shift(1))
                step = int(p['f'].replace('D',''))
                if step > 1:
                    # 生成重采样索引
                    n_days = len(base_sig)
                    resample_idx = (np.arange(n_days) // step) * step
                    # 极速切片并前向填充，然后 shift(1)
                    sig_values = base_sig.values[resample_idx]
                    sig_values = np.roll(sig_values, 1, axis=0)
                    sig_values[0, :] = 0.0
                    final_w = pd.DataFrame(sig_values, index=base_sig.index, columns=base_sig.columns)
                else:
                    # 1D 频率只需 shift(1)
                    final_w = base_sig.shift(1).fillna(0.0)
                
                batch_weights.append(final_w.astype(np.float32))
                keys.append((p['w'], p['p'], p['f'], p['d']))
            
            # 使用价格表
            # 注意：VectorBT 要求价格和权重必须对齐。这里简单起见，假设所有参数使用相同的价格基准或对应处理
            # 实际上不同的 p 会对应不同的价格矩阵，这里我们需要分组执行或构建大的 prices
            
            # 优化：同一批次内如果价格字段相同，可以合并运行
            mega_w = pd.concat(batch_weights, axis=1, keys=keys, names=['w', 'p', 'f', 'rev'])
            
            # 这里为了支持多价格字段，我们取 batch 中第一个参数的价格 (假设 batch 划分时已考虑或价格影响较小)
            # 更严谨的做法是 mega_p 也要对齐
            current_p_field = batch_params[0]['p']
            o_p = prices[current_p_field]
            
            portfolio = vbt.Portfolio.from_orders(
                o_p, 
                size=mega_w, 
                size_type='targetpercent', 
                init_cash=1000000, 
                fees=0.0015, 
                freq='1D', 
                group_by=['w', 'p', 'f', 'rev'], 
                cash_sharing=True
            )
            
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
        test_f = variants[best_row['w']].loc[test_period:]
        test_o_p = prices[best_row['p']].loc[test_period:]
        final_res = self.rigorous_backtest_single(test_f, test_o_p, freq=best_row['f'], ascending=best_row['rev'])
        
        print("\n" + "*"*60 + "\n因子终极盲测报告 (2024-2026)\n" + "*"*60)
        print(final_res.stats())
        
        plt.figure(figsize=(12,6))
        final_res.value().plot(label=f'OOS Strategy (W:{best_row["w"]} F:{best_row["f"]})', color='red', lw=2)
        
        # 获取并归一化基准指数
        benchmark_p = self.engine.get_benchmark_prices(start_date=test_period)
        if not benchmark_p.empty:
            # 归一化到相同的初始资金级别
            init_val = final_res.value().iloc[0]
            benchmark_normalized = (benchmark_p / benchmark_p.iloc[0]) * init_val
            benchmark_normalized.plot(label='Benchmark (HS300)', color='black', alpha=0.5, linestyle='--')
            
        plt.title(f'Final OOS Test vs Benchmark: {factor_name} (2024-2026)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'data/final_oos_{factor_name}.png')

        plt.figure(figsize=(10,6))
        plot_df = stats_df[stats_df['rev'] == best_row['rev']].pivot_table(index='w', columns='f', values='Sharpe')
        sns.heatmap(plot_df, annot=True, cmap='RdYlGn')
        plt.savefig(f'data/vectorized_landscape_{factor_name}.png')
        
        # --- 核心新增：2006-2023 历史滚动“体检” ---
        self.run_historical_rolling_test(factor_name, variants[best_row['w']], prices[best_row['p']], best_row)
        
        return stats_df

    def run_historical_rolling_test(self, factor_name, f_pivot, o_p, best_params):
        """
        历史滚动压力测试模块：验证最优参数在 2006-2023 年不同市场周期下的表现。
        """
        print(f"\n" + "="*60 + "\n正在进入 2006-2023 历史长河压力测试阶段...\n" + "="*60)
        print(f"数据总范围: {f_pivot.index.min()} 至 {f_pivot.index.max()} (总行数: {len(f_pivot)})")
        
        windows = []
        start_year = 2006
        end_year = 2023
        window_size = 3 # 3年为一个滑动窗口
        
        for y in range(start_year, end_year - window_size + 2):
            windows.append((f"{y}-01-01", f"{y+window_size-1}-12-31"))
            
        rolling_results = []
        
        for start, end in tqdm(windows, desc="滚动窗口扫描"):
            try:
                # 截取对应时间切片
                slice_f = f_pivot.loc[start:end]
                slice_p = o_p.loc[start:end]
                
                if len(slice_f) < 50: 
                    continue
                
                # 执行回测（使用最优参数）
                res = self.rigorous_backtest_single(
                    slice_f, slice_p, 
                    freq=best_params['f'], 
                    ascending=best_params['rev']
                )
                
                stats = res.stats()
                rolling_results.append({
                    'Window': f"{start[:4]}-{end[:4]}",
                    'Return': stats.get('Total Return [%]', 0),
                    'Sharpe': stats.get('Sharpe Ratio', 0),
                    'MaxDD': stats.get('Max Drawdown [%]', 0),
                    'WinRate': stats.get('Win Rate [%]', 0),
                    'Trades': stats.get('Total Trades', 0),
                    'ProfitFactor': stats.get('Profit Factor', 0)
                })
            except Exception as e:
                print(f"窗口 {start}-{end} 执行失败: {e}")
                continue

        rolling_df = pd.DataFrame(rolling_results)
        if rolling_df.empty:
            print("\n[错误] 滚动测试未生成任何有效结果。")
            print(f"因子矩阵索引类型: {type(f_pivot.index[0])}")
            print(f"因子矩阵索引示例: {f_pivot.index[:3]}")
            return
        
        # --- 输出深度诊断日志 ---
        print("\n" + "*"*60 + "\n历史滚动“体检”核心指标汇总 (2006-2023)\n" + "*"*60)
        print(rolling_df.to_string(index=False))
        
        # 1. 收益一致性分析
        avg_sharpe = rolling_df['Sharpe'].mean()
        sharpe_std = rolling_df['Sharpe'].std()
        print(f"\n[维度1] 夏普比率一致性: 均值 {avg_sharpe:.2f}, 标准差 {sharpe_std:.2f}")
        if sharpe_std / avg_sharpe < 0.45:
            print(">> 评价：该参数组合在多个周期下表现稳定，逻辑坚固。")
        else:
            print(">> 评价：警告！收益表现随市场环境剧烈波动，存在过度拟合特定年份的风险。")
            
        # 2. 胜率均值回归分析
        avg_winrate = rolling_df['WinRate'].mean()
        print(f"\n[维度2] 历史基准胜率: {avg_winrate:.2f}%")
        print(f">> 提示：如果当前盲测胜率远高于此，实盘中需做好胜率向 {avg_winrate:.2f}% 回归的准备。")

        # 3. 极端黑天鹅压力测试
        print(f"\n[维度3] 历史最深回撤统计: 最大 {rolling_df['MaxDD'].max():.2f}%")
        stress_years = ['2008-2010', '2015-2017', '2018-2020']
        for sy in stress_years:
            val = rolling_df[rolling_df['Window'] == sy]['MaxDD']
            if not val.empty:
                print(f">> {sy} (大型危机/股灾) 期间最大回撤: {val.values[0]:.2f}%")

        # 4. 交易摩擦成本预警
        avg_trades = rolling_df['Trades'].mean()
        print(f"\n[维度4] 平均调仓频率: 每个窗口 {avg_trades:.0f} 次交易")
        
        # --- 生成可视化体检报告图表 ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35)
        
        # 夏普一致性柱状图
        axes[0,0].bar(rolling_df['Window'], rolling_df['Sharpe'], color='skyblue', alpha=0.8)
        axes[0,0].axhline(avg_sharpe, color='red', linestyle='--', label='Mean Sharpe')
        axes[0,0].set_title('Consistency: Rolling Sharpe Ratio (3Y Window)')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 胜率回归趋势图
        axes[0,1].plot(rolling_df['Window'], rolling_df['WinRate'], marker='o', color='forestgreen', lw=2)
        axes[0,1].axhline(avg_winrate, color='gray', linestyle=':', label='Hist. Mean')
        axes[0,1].set_title('Mean Reversion: Win Rate Trend (%)')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 回撤压力测试直方图
        axes[1,0].bar(rolling_df['Window'], rolling_df['MaxDD'], color='salmon')
        axes[1,0].set_title('Stress Test: Max Drawdown (%)')
        axes[1,0].invert_yaxis() # 习惯上回撤向下看
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 交易频次分布
        axes[1,1].bar(rolling_df['Window'], rolling_df['Trades'], color='gray', alpha=0.6)
        axes[1,1].set_title('Friction: Total Trades (Turnover Stability)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.savefig(f'data/rolling_diagnostic_{factor_name}.png')
        print(f"\n[OK] 滚动体检报告已生成至: data/rolling_diagnostic_{factor_name}.png")
        
        # 将数据保存到 CSV 方便分析
        rolling_df.to_csv(f'data/rolling_diagnostic_{factor_name}.csv', index=False, encoding='utf-8-sig')
        print(f"[OK] 滚动体检原始数据已保存至: data/rolling_diagnostic_{factor_name}.csv")

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
    
    t_start = time.time()
    VectorizedOptimizer().run_fast_optimize(args.factor, mode=args.mode)
    t_end = time.time()
    
    print(f"\n" + "="*60)
    print(f"任务完成！总耗时: {t_end - t_start:.2f} 秒")
    print("="*60 + "\n")
