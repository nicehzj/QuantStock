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
    """
    
    def __init__(self):
        self.engine = FactorEngine()
        self.connector = DBConnector()

    def get_forward_returns(self, periods=[1, 5, 10]):
        print(f">>> 正在计算未来收益率 (Periods: {periods})...")
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
        factors_df = self.engine.calculate_basic_factors()
        prices_raw = self.get_forward_returns()
        
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        prices_raw['date'] = pd.to_datetime(prices_raw['date'])
        
        factors_df = factors_df[(factors_df['date'] >= start_date) & (factors_df['date'] <= end_date)]
        prices_raw = prices_raw[(prices_raw['date'] >= start_date) & (prices_raw['date'] <= pd.to_datetime(end_date) + pd.Timedelta(days=30))]

        if factors_df.empty:
            print(f"[Warning] 警告: 指定日期范围 {start_date} -> {end_date} 内没有因子数据。")
            return None, factors_df

        f_series = factors_df.set_index(['date', 'code'])[factor_name].sort_index()
        prices_pivot = prices_raw.pivot(index='date', columns='code', values='close')
        prices_pivot.index = pd.to_datetime(prices_pivot.index)
        prices_pivot = prices_pivot.ffill().bfill()

        print(f">>> 正在运行 Alphalens 统计分析: {factor_name}")
        try:
            merged_data = alphalens.utils.get_clean_factor_and_forward_returns(
                factor=f_series,
                prices=prices_pivot,
                periods=[1, 5, 10],
                quantiles=5,
                max_loss=0.4 
            )
            ic = alphalens.performance.factor_information_coefficient(merged_data)
            print("\n" + "="*40)
            print(f"因子 [{factor_name}] 的 IC 统计概览:")
            print(ic.describe())
            print("="*40)
            mean_return, _ = alphalens.performance.mean_return_by_quantile(merged_data)
            print("\n各分位数组平均未来 5 日收益率:")
            print(mean_return['5D'])
            return merged_data, factors_df
        except Exception as e:
            print(f"[Error] Alphalens 分析出错: {e}")
            return None, factors_df

    def push_signals_to_redis(self, factor_df, factor_name='factor_mom_20', top_n=50):
        r = self.connector.get_redis()
        if not r:
            print("[Warning] Redis 未连接，跳过信号推送。")
            return
        
        ldate, snapshot = self.engine.get_latest_market_snapshot(factor_df)
        top_signals = snapshot.sort_values(factor_name, ascending=False).head(top_n)
        
        if not top_signals.empty:
            signal_key = f"signals:{ldate}"
            r.delete(signal_key)
            codes = top_signals['code'].tolist()
            r.rpush(signal_key, *codes)
            scores_key = f"scores:{ldate}"
            score_dict = {row['code']: row[factor_name] for _, row in top_signals.iterrows()}
            r.hset(scores_key, mapping=score_dict)
            r.expire(signal_key, 86400 * 30)
            r.expire(scores_key, 86400 * 30)
            print(f"[OK] 已将 {ldate} 的 {len(codes)} 条交易信号推送到 Redis。")

    def export_historical_signals(self, factor_name='factor_mom_20', top_n=20, start_date='2021-01-01', filename='data/historical_signals.csv', ascending=True):
        factors_df = self.engine.calculate_basic_factors()
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        oos_factors = factors_df[factors_df['date'] >= start_date].copy()
        print(f">>> 正在生成历史选股信号 (Top {top_n}, {'超跌反转' if ascending else '趋势跟踪'})...")
        oos_factors['rank'] = oos_factors.groupby('date')[factor_name].rank(ascending=ascending, method='first')
        signals = oos_factors[oos_factors['rank'] <= top_n][['date', 'code', factor_name]]
        signals.to_csv(filename, index=False)
        print(f"[OK] 历史信号已导出至: {filename} (共 {len(signals)} 条记录)")
        return signals

if __name__ == "__main__":
    evaluator = AlphaEvaluator()
    analysis_res, factors_raw = evaluator.run_alphalens_analysis('factor_mom_20')
    evaluator.export_historical_signals(top_n=20)
    if analysis_res is not None:
        evaluator.push_signals_to_redis(factors_raw)
