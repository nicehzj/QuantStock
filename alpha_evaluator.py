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

    def run_alphalens_analysis(self, factor_name='factor_mom_20'):
        """
        运行 Alphalens 分析。
        它会自动处理因子的分位数切分、收益对齐和 IC 计算。
        """
        # 1. 提取因子数据
        factors_df = self.engine.calculate_basic_factors()
        
        # 2. 提取价格数据（用于计算未来收益）
        prices_raw = self.get_forward_returns()
        
        # 3. 准备 Alphalens 格式的因子数据 (MultiIndex: [date, asset])
        factors_df['date'] = pd.to_datetime(factors_df['date'])
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

if __name__ == "__main__":
    # 创建评估实例
    evaluator = AlphaEvaluator()
    
    # 执行评估流程：动量因子评估
    analysis_res, factors_raw = evaluator.run_alphalens_analysis('factor_mom_20')
    
    # 如果评估通过，将信号持久化到 Redis
    if analysis_res is not None:
        evaluator.push_signals_to_redis(factors_raw)
    else:
        print("💡 评估数据不足，无法生成有效信号。请检查本地 Parquet 数据是否覆盖了足够长的时间范围。")
