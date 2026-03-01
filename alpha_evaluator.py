import pandas as pd
import numpy as np
import alphalens
from factor_engine import FactorEngine
from db_connector import DBConnector
import matplotlib.pyplot as plt
from datetime import datetime

class AlphaEvaluator:
    """
    因子评估专家
    负责将计算出的因子与未来收益率对齐，并利用 Alphalens 进行因子有效性分析。
    """
    
    def __init__(self):
        # 初始化因子计算引擎
        self.engine = FactorEngine()
        # 初始化数据库连接器
        self.connector = DBConnector()

    def get_forward_returns(self, periods=[1, 5, 10]):
        """
        利用 DuckDB 批量计算全市场股票的未来 N 日收益率。
        这是 Alphalens 评估的核心输入。
        """
        print(f">>> 正在计算未来收益率 (Periods: {periods})...")
        
        # 动态构造 SQL：使用 LEAD 窗口函数获取未来价格
        lead_sqls = [
            f"(LEAD(close, {p}) OVER(PARTITION BY code ORDER BY date) / close - 1) as fwd_ret_{p}"
            for p in periods
        ]
        
        sql = f"""
        SELECT 
            CAST(date AS DATE) as date,
            code,
            close,
            {', '.join(lead_sqls)}
        FROM read_parquet('{self.engine.parquet_glob}')
        ORDER BY date, code
        """
        return self.engine.db.execute(sql).df()

    def run_alphalens_analysis(self, factor_name='factor_mom_20', start_date='2006-01-01', end_date='2020-12-31'):
        """
        运行 Alphalens 分析。
        它会自动处理因子的分位数切分、收益对齐和 IC 计算。
        """
        # 1. 提取因子数据
        factors_df = self.engine.calculate_basic_factors()
        
        # 2. 提取价格数据（用于计算未来收益）
        prices_raw = self.get_forward_returns()
        
        # 3. 日期过滤：仅保留样本内数据 (In-Sample)
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        prices_raw['date'] = pd.to_datetime(prices_raw['date'])
        
        factors_df = factors_df[(factors_df['date'] >= start_date) & (factors_df['date'] <= end_date)]
        # 价格数据需要稍微多一点点，以满足 Alphalens 内部对未来收益计算的对齐（虽然我们在 SQL 里算好了，但 Alphalens 还需要价格矩阵）
        prices_raw = prices_raw[(prices_raw['date'] >= start_date) & (prices_raw['date'] <= pd.to_datetime(end_date) + pd.Timedelta(days=30))]

        if factors_df.empty:
            print(f"⚠️ 警告: 指定日期范围 {start_date} -> {end_date} 内没有因子数据。")
            return None, factors_df

        # 4. 准备 Alphalens 格式的因子数据 (MultiIndex: [date, asset])
        # 确保索引 unique 并排序
        f_series = factors_df.set_index(['date', 'code'])[factor_name].sort_index()
        
        # 4. 准备 Alphalens 格式的价格矩阵 (Rows: date, Cols: assets)
        prices_pivot = prices_raw.pivot(index='date', columns='code', values='close')
        prices_pivot.index = pd.to_datetime(prices_pivot.index)
        # 填充缺失值（Alphalens 对价格矩阵的完整性有要求）
        prices_pivot = prices_pivot.ffill().bfill()

        print(f">>> 正在运行 Alphalens 统计分析: {factor_name}")
        
        try:
            # 5. 获取 Alphalens 内部格式的数据集
            # quantiles=5 代表将所有股票按因子值均分为 5 组进行对比
            merged_data = alphalens.utils.get_clean_factor_and_forward_returns(
                factor=f_series,
                prices=prices_pivot,
                periods=[1, 5, 10],
                quantiles=5,
                max_loss=0.4 # 允许 40% 的数据丢失（由于退市、停牌或首尾日期）
            )
            
            # 6. 计算 IC 值 (Information Coefficient)
            # IC 衡量因子值与未来收益的相关性，是评估因子预测能力的核心指标
            ic = alphalens.performance.factor_information_coefficient(merged_data)
            print("\n" + "="*40)
            print(f"因子 [{factor_name}] 的 IC 统计概览:")
            print(ic.describe())
            print("="*40)
            
            # 7. 计算各分位数的平均收益率
            # 理想情况下，第 5 组（最高值组）的收益应远高于第 1 组
            mean_return, _ = alphalens.performance.mean_return_by_quantile(merged_data)
            print("\n各分位数组平均未来 5 日收益率:")
            print(mean_return['5D'])
            
            return merged_data, factors_df
            
        except Exception as e:
            print(f"❌ Alphalens 分析出错: {e}")
            return None, factors_df

    def push_signals_to_redis(self, factor_df, factor_name='factor_mom_20', top_n=50):
        """
        信号分发：将每日评分最高的股票推送到 Redis。
        这实现了计算层与回测/执行层的解耦。
        """
        r = self.connector.get_redis()
        if not r:
            print("⚠️ Redis 未连接，跳过信号推送。")
            return
        
        # 获取最新一个交易日的截面数据
        ldate, snapshot = self.engine.get_latest_market_snapshot(factor_df)
        
        # 筛选因子值最强的 Top N 只股票作为买入信号
        top_signals = snapshot.sort_values(factor_name, ascending=False).head(top_n)
        
        if not top_signals.empty:
            # 信号存入 Redis Key: "signals:YYYY-MM-DD"
            signal_key = f"signals:{ldate}"
            # 先清空旧信号
            r.delete(signal_key)
            # 推送代码列表
            codes = top_signals['code'].tolist()
            r.rpush(signal_key, *codes)
            
            # 同时在 Redis 中存储一份带有分数的 Hash，方便查看每个股票的因子值
            scores_key = f"scores:{ldate}"
            score_dict = {row['code']: row[factor_name] for _, row in top_signals.iterrows()}
            r.hmset(scores_key, score_dict)
            
            # 设置过期时间（保持 30 天）
            r.expire(signal_key, 86400 * 30)
            r.expire(scores_key, 86400 * 30)
            
            print(f"✅ 已将 {ldate} 的 {len(codes)} 条交易信号推送到 Redis。")
            print(f"   - 信号列表: {signal_key}")
            print(f"   - 因子分值: {scores_key}")

    def export_historical_signals(self, factor_name='factor_mom_20', top_n=20, start_date='2021-01-01', filename='data/historical_signals.csv'):
        """
        导出历史选股信号表。
        用于驱动精细化回测引擎进行全市场模拟。
        """
        # 1. 计算因子
        factors_df = self.engine.calculate_basic_factors()
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        
        # 2. 筛选样本外数据
        oos_factors = factors_df[factors_df['date'] >= start_date].copy()
        
        print(f">>> 正在生成历史选股信号 (Top {top_n})...")
        
        # 3. 逐日筛选排名最高的股票
        # 使用 rank 函数快速处理截面排名
        oos_factors['rank'] = oos_factors.groupby('date')[factor_name].rank(ascending=False)
        signals = oos_factors[oos_factors['rank'] <= top_n][['date', 'code', factor_name]]
        
        # 4. 保存到本地
        signals.to_csv(filename, index=False)
        print(f"✅ 历史信号已导出至: {filename} (共 {len(signals)} 条记录)")
        return signals

if __name__ == "__main__":
    # 创建评估实例
    evaluator = AlphaEvaluator()
    
    # 步骤 A: 运行样本内评估 (2006-2020)
    analysis_res, factors_raw = evaluator.run_alphalens_analysis('factor_mom_20')
    
    # 步骤 B: 导出样本外历史信号 (2021-Present), 用于精细回测
    evaluator.export_historical_signals(top_n=20)
    
    # 如果评估通过，将最新信号推送
    if analysis_res is not None:
        evaluator.push_signals_to_redis(factors_raw)
