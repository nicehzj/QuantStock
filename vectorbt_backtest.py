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
            print(f"⚠️ 错误: 在区间 {start_date} -> {end_date} 内没有足够的回测数据。")
            return None

        # 处理缺失值（退市或停牌股票），保持矩阵完整性
        price_df = price_df.ffill()

        # 4. 生成交易信号 (Entries)
        print(f">>> 正在根据动量因子生成 Top {top_n} 信号矩阵...")
        # 将因子 DataFrame 转换为与价格矩阵对齐的宽表
        factor_pivot = factors_df.pivot(index='date', columns='code', values='factor_mom_20')
        factor_pivot.index = pd.to_datetime(factor_pivot.index)
        
        # --- 核心修复：强制对齐所有矩阵 ---
        # 找到两个矩阵共同的日期和股票代码
        common_index = price_df.index.intersection(factor_pivot.index)
        common_columns = price_df.columns.intersection(factor_pivot.columns)
        
        price_df = price_df.loc[common_index, common_columns]
        factor_pivot = factor_pivot.loc[common_index, common_columns]
        
        # 对每日截面进行排名，筛选前 top_n
        entries = factor_pivot.rank(axis=1, ascending=False) <= top_n
        # 调仓逻辑：今日不在 Top N 列表中的股票全部卖出 (Exits)
        exits = ~entries
        
        # 填充价格矩阵中的 NaN（Vectorbt 要求价格必须连续且无 NaN）
        price_df = price_df.ffill().bfill()
        
        # 4. 执行向量化回测
        print(f">>> Vectorbt 引擎正在并行计算收益率曲线 (标的数: {len(common_columns)})...")
        # cash_sharing=True 是关键：它模拟了一个统一的现金池在多只股票间分配
        portfolio = vbt.Portfolio.from_signals(
            price_df, 
            entries=entries, 
            exits=exits, 
            freq='1D',
            init_cash=100000,
            fees=fees,
            cash_sharing=True,
            group_by=True # 将所有结果聚合为一个投资组合
        )
        
        duration = time.time() - start_time
        print(f"✅ 全市场回测完成！总耗时: {duration:.2f} 秒")
        
        # 5. 打印核心统计指标
        stats = portfolio.stats()
        print("\n" + "="*50)
        print("Vectorbt 全市场动量策略统计报告")
        print("-" * 50)
        # 兼容不同版本的 Key 名称
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
        # 运行回测：每日持有动量最强的 30 只股票
        portfolio = tester.run_momentum_screening(top_n=30)
        
        # 提示：如果需要绘图，可以调用 portfolio.plot()
        # 注意在 CLI 环境下可能需要导出图片：portfolio.plot().write_image("vbt_result.png")
    except Exception as e:
        print(f"❌ 回测运行失败: {e}")
