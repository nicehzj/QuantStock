import backtrader as bt
import pandas as pd
import os
import duckdb
import numpy as np
from datetime import datetime

class AShareCommission(bt.CommInfoBase):
    params = (
        ('stamp_duty', 0.001),      
        ('commission', 0.0003),     
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )
    def _getcommission(self, size, price, pseudoexec):
        if size > 0: return size * price * self.p.commission
        elif size < 0: return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        return 0

class CustomPandasData(bt.feeds.PandasData):
    lines = ('preclose',)
    params = (('preclose', -1),)

class SignalStrategy(bt.Strategy):
    params = (('signals', None), ('top_n', 20), ('verbose', False))

    def log(self, txt, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.buy_dates = {}
        self.daily_signals = self.p.signals.groupby('date')['code'].apply(list).to_dict()
        self.orders = {} 
        self.audit_records = [] 
        self.data_map = {d._name: d for d in self.datas}
        self.current_pos_set = set()
        self.pending_sell_audits = {}

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.current_pos_set.add(order.data._name)
            else:
                if order.data._name in self.current_pos_set:
                    self.current_pos_set.remove(order.data._name)
            self.orders[order.data._name] = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def notify_trade(self, trade):
        if trade.isclosed:
            code = trade.data._name
            if code in self.pending_sell_audits:
                idx = self.pending_sell_audits[code]
                if idx < len(self.audit_records):
                    rec = self.audit_records[idx]
                    rec['PnL'] = round(trade.pnl, 2)
                    rec['PnL_Pct'] = round(trade.pnlnew / trade.value * 100, 2) if trade.value != 0 else 0
                del self.pending_sell_audits[code]

    def next(self):
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        target_codes = self.daily_signals.get(dt_ts, [])

        # 1. 卖出逻辑
        to_sell = [c for c in self.current_pos_set if c not in target_codes]
        for code in to_sell:
            d = self.data_map.get(code)
            if not d: continue
            bdt = self.buy_dates.get(code)
            if not (bdt and dt > bdt): continue
            
            lr = 0.199 if (code.startswith('sz.30') or code.startswith('sh.688')) else 0.099
            if d.close[0] <= d.preclose[0] * (1 - lr):
                self.audit_records.append({'Timestamp': dt, 'Code': code, 'Action': 'SELL', 'Price': d.close[0], 'Status': 'Failed', 'Reason': 'Limit Down', 'Size': self.getposition(d).size, 'PnL': 0, 'PnL_Pct': 0})
                continue
            
            if not self.orders.get(code):
                self.pending_sell_audits[code] = len(self.audit_records)
                self.audit_records.append({'Timestamp': dt, 'Code': code, 'Action': 'SELL', 'Price': d.close[0], 'Status': 'Success', 'Reason': '', 'Size': self.getposition(d).size, 'PnL': 0, 'PnL_Pct': 0})
                self.orders[code] = self.close(data=d)

        # 2. 买入逻辑
        if target_codes:
            to_buy = [c for c in target_codes if c not in self.current_pos_set]
            budget = (self.broker.get_value() * 0.95) / self.p.top_n
            for code in to_buy:
                if self.orders.get(code): continue
                d = self.data_map.get(code)
                if not d: continue
                lr = 0.199 if (code.startswith('sz.30') or code.startswith('sh.688')) else 0.099
                if d.close[0] >= d.preclose[0] * (1 + lr):
                    self.audit_records.append({'Timestamp': dt, 'Code': code, 'Action': 'BUY', 'Price': d.close[0], 'Status': 'Failed', 'Reason': 'Limit Up', 'Size': 0, 'PnL': 0, 'PnL_Pct': 0})
                    continue
                size = int(budget / d.close[0] / 100) * 100
                if size >= 100:
                    self.orders[code] = self.buy(data=d, size=size)
                    self.buy_dates[code] = dt
                    self.audit_records.append({'Timestamp': dt, 'Code': code, 'Action': 'BUY', 'Price': d.close[0], 'Status': 'Success', 'Reason': '', 'Size': size, 'PnL': 0, 'PnL_Pct': 0})
                else:
                    self.audit_records.append({'Timestamp': dt, 'Code': code, 'Action': 'BUY', 'Price': d.close[0], 'Status': 'Failed', 'Reason': 'No Cash', 'Size': 0, 'PnL': 0, 'PnL_Pct': 0})

class TradeLogger(bt.Analyzer):
    def __init__(self): self.trades = []
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({'code': trade.data._name, 'pnl': trade.pnl})
    def get_analysis(self): return self.trades

def run_full_market_backtest(signal_file='data/historical_signals.csv', start_date='2021-01-01'):
    if not os.path.exists(signal_file): return
    signals_df = pd.read_csv(signal_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    stock_bounds = signals_df.groupby('code')['date'].agg(['min', 'max']).reset_index()
    stock_bounds.columns = ['code', 'first_sig', 'last_sig']
    
    target_stocks = list(stock_bounds['code'].unique())
    print(f">>> 正在利用 DuckDB 加载 {len(target_stocks)} 只股票数据...")
    
    db = duckdb.connect(database=':memory:')
    db.register('bounds', stock_bounds)
    parquet_glob = "data/daily_k/*.parquet"
    idx_path = "data/index_k/sh.000001.parquet"
    
    idx_df = db.execute(f"SELECT CAST(date AS TIMESTAMP) as date, open, high, low, close, volume, preclose FROM read_parquet('{idx_path}') WHERE date >= '{start_date}'").df()
    idx_df.set_index('date', inplace=True)
    
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(CustomPandasData(dataname=idx_df), name='Benchmark_Index')

    sql = f"SELECT p.code, CAST(p.date AS TIMESTAMP) as date, p.open, p.high, p.low, p.close, p.volume, p.preclose FROM read_parquet('{parquet_glob}') p JOIN bounds b ON p.code = b.code WHERE CAST(p.date AS TIMESTAMP) >= '{start_date}'"
    all_data = db.execute(sql).df()
    
    for code, group in all_data.groupby('code'):
        group = group.sort_values('date').set_index('date').reindex(idx_df.index).ffill().bfill()
        cerebro.adddata(CustomPandasData(dataname=group), name=code)
    
    cerebro.addstrategy(SignalStrategy, signals=signals_df)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.addcommissioninfo(AShareCommission())
    cerebro.addanalyzer(TradeLogger, _name='tradelog')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f'>>> 回测启动 (2021-至今)')
    results = cerebro.run(runonce=True, preload=True)
    strat = results[0]

    if strat.audit_records:
        pd.DataFrame(strat.audit_records).to_csv('data/bt_unified_trade_log.csv', index=False)
        print(f"[OK] 统一交易日志已生成: data/bt_unified_trade_log.csv")

    final_cash = cerebro.broker.get_cash()
    current_mv = sum([pos.size * d.close[0] for d, pos in cerebro.broker.positions.items() if pos.size > 0 and not np.isnan(d.close[0])])
    final_total = final_cash + current_mv
    print(f"\n账户总资产: {final_total:,.2f} | 收益率: {(final_total/1000000.0-1)*100:.2f}%")
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"最大回撤: {dd.max.drawdown:.2f}% | 交易数: {strat.analyzers.trades.get_analysis().total.total}")
    except: pass

if __name__ == "__main__":
    run_full_market_backtest()
