import backtrader as bt
import pandas as pd
import os
import duckdb
from datetime import datetime

class AShareCommission(bt.CommInfoBase):
    """
    A 股专用佣金与印花税模型
    """
    params = (
        ('stamp_duty', 0.001),      # 卖出印花税 0.1%
        ('commission', 0.0003),     # 交易佣金 0.03%
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0: # 买入
            return size * price * self.p.commission
        elif size < 0: # 卖出
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        return 0

class CustomPandasData(bt.feeds.PandasData):
    """
    自定义 Pandas 数据源，显式映射 preclose 字段
    """
    lines = ('preclose',)
    params = (('preclose', -1),) # 自动按名称匹配

class SignalStrategy(bt.Strategy):
    """
    全市场信号同步策略 - 极致严谨 A 股版
    1. 信号日收盘涨停 -> 次日不可买
    2. 信号日收盘跌停 -> 次日不可卖
    3. 严格 T+1
    4. 先卖后买，释放资金
    """
    params = (
        ('signals', None), 
        ('verbose', True),
    )

    def log(self, txt, dt=None):
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.buy_dates = {}
        # 预处理信号
        self.daily_signals = self.p.signals.groupby('date')['code'].apply(list).to_dict()
        self.orders = {} 

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入成交: {order.data._name}, 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}')
            else:
                self.log(f'卖出成交: {order.data._name}, 价格: {order.executed.price:.2f}')
            self.orders[order.data._name] = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.orders[order.data._name] = None

    def next(self):
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        target_codes = self.daily_signals.get(dt_ts, [])

        # --- 第一步：处理卖出逻辑 (释放资金) ---
        curr_positions = [d._name for d, pos in self.getpositions().items() if pos.size > 0]
        for code in curr_positions:
            if code not in target_codes:
                buy_dt = self.buy_dates.get(code)
                if buy_dt and dt > buy_dt: # T+1
                    data = self.getdatabyname(code)
                    if data.close[0] <= data.preclose[0] * 0.901: # 跌停卖不出
                        self.log(f'⚠️ {code} 封死跌停，今日无法建立卖单')
                        continue
                    if not self.orders.get(code):
                        self.orders[code] = self.close(data=data)

        # --- 第二步：处理买入逻辑 ---
        if target_codes:
            to_buy = [code for code in target_codes if not self.getpositionbyname(code).size > 0]
            if to_buy:
                total_value = self.broker.get_value()
                # 资金分配：总资产 90% 平分给 Top 20
                per_stock_budget = (total_value * 0.90) / 20 # 核心修改：Top 20 分散化
                
                for code in to_buy:
                    if self.orders.get(code): continue
                    try:
                        data = self.getdatabyname(code)
                        if data.close[0] >= data.preclose[0] * 1.099: # 涨停买不进
                            self.log(f'🚫 {code} 封死涨停，今日无法建立买单')
                            continue
                        price = data.close[0]
                        if price > 0:
                            size = int(per_stock_budget / price / 100) * 100
                            if size > 0:
                                self.orders[code] = self.buy(data=data, size=size)
                                self.buy_dates[code] = dt
                    except Exception: continue

class TradeLogger(bt.Analyzer):
    """
    自定义分析器：记录每一笔交易的详细明细
    """
    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'code': trade.data._name,
                'entry_date': bt.num2date(trade.dtopen).isoformat(),
                'exit_date': bt.num2date(trade.dtclose).isoformat(),
                'pnl': round(trade.pnl, 2),
                'pnl_pct': round(trade.pnlnew / trade.value * 100, 2) if trade.value != 0 else 0,
                'size': trade.size,
                'value': round(trade.value, 2)
            })

    def get_analysis(self):
        return self.trades

def run_full_market_backtest(signal_file='data/historical_signals.csv', start_date='2021-01-01'):
    """
    极速版全市场回测引擎 (活跃窗口片段优化)
    """
    if not os.path.exists(signal_file):
        print(f"❌ 未找到信号文件: {signal_file}")
        return

    # 1. 读取信号并计算每只股票的活跃窗口
    signals_df = pd.read_csv(signal_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # 统计每只股票的 min/max 信号日期
    stock_bounds = signals_df.groupby('code')['date'].agg(['min', 'max']).reset_index()
    stock_bounds.columns = ['code', 'first_sig', 'last_sig']
    
    target_stocks = list(stock_bounds['code'].unique())

    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(False)
    
    # 2. 利用 DuckDB 提取“数据片段”
    print(f">>> 正在利用 DuckDB 精准拉取 {len(target_stocks)} 只股票的活跃片段...")
    db = duckdb.connect(database=':memory:')
    
    # 注册 bounds 表供 SQL 使用
    db.register('bounds', stock_bounds)
    
    parquet_glob = "D:/MyCode/QuantStock/data/daily_k/*.parquet"
    idx_path = "D:/MyCode/QuantStock/data/index_k/sh.000001.parquet"
    
    # 加载基准指数 (完整时间轴)
    idx_df = db.execute(f"SELECT CAST(date AS TIMESTAMP) as date, open, high, low, close, volume, preclose FROM read_parquet('{idx_path}') WHERE date >= '{start_date}'").df()
    idx_df.set_index('date', inplace=True)
    cerebro.adddata(CustomPandasData(dataname=idx_df), name='Benchmark_Index')

    # 核心 SQL 优化：JOIN 过滤，只取活跃日期前 5 天到后 10 天的片段
    sql = f"""
    SELECT 
        p.code,
        CAST(p.date AS TIMESTAMP) as date,
        p.open, p.high, p.low, p.close, p.volume, p.preclose
    FROM read_parquet('{parquet_glob}') p
    JOIN bounds b ON p.code = b.code
    WHERE CAST(p.date AS TIMESTAMP) >= (b.first_sig - INTERVAL 5 DAYS) 
      AND CAST(p.date AS TIMESTAMP) <= (b.last_sig + INTERVAL 10 DAYS)
      AND CAST(p.date AS TIMESTAMP) >= CAST('{start_date}' AS TIMESTAMP)
    ORDER BY p.date
    """
    fragment_data_df = db.execute(sql).df()
    
    print(f">>> 注入 Backtrader 数据片段 (总行数: {len(fragment_data_df)}，原约 110 万行)...")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'preclose']
    
    for code, group in fragment_data_df.groupby('code'):
        group = group.sort_values('date')
        group[numeric_cols] = group[numeric_cols].astype(float).ffill().bfill()
        
        if len(group) < 2: continue # 过滤掉数据太短的
            
        group = group.set_index('date')
        data = CustomPandasData(dataname=group)
        cerebro.adddata(data, name=code)
    
    print(f"✅ 成功加载 {len(target_stocks)} 只股票的活跃片段。")

    # 3. 运行回测
    cerebro.addstrategy(SignalStrategy, signals=signals_df)
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.addcommissioninfo(AShareCommission())
    # 4. 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(TradeLogger, _name='tradelog')

    print(f'>>> 回测启动 (样本外: {start_date} 至今)')
    results = cerebro.run()
    strat = results[0]

    # --- 核心新增：导出交易明细 ---
    trade_list = strat.analyzers.tradelog.get_analysis()
    if trade_list:
        trades_df = pd.DataFrame(trade_list)
        trades_df.to_csv('data/backtest_trades.csv', index=False)
        print(f"✅ 交易明细已导出至: data/backtest_trades.csv (共 {len(trades_df)} 笔已平仓交易)")
    else:
        print("⚠️ 未产生任何已平仓交易明细。")

    # --- 5. 输出详细统计结果 ---
    final_total = cerebro.broker.getvalue()
    final_cash = cerebro.broker.get_cash()
    market_value = final_total - final_cash

    print("\n" + "="*50)
    print("QuantStock 极速版回测资产报告")
    print("-" * 50)
    print(f"账户总资产: {final_total:,.2f}")
    print(f"可用现金:   {final_cash:,.2f} ({final_cash/final_total*100:.1f}%)")
    print(f"持仓市值:   {market_value:,.2f} ({market_value/final_total*100:.1f}%)")
    print("-" * 50)
    print(f"累计收益率: {(final_total/1000000.0 - 1)*100:.2f}%")
    
    # 打印当前持仓明细 (未平仓位)
    open_positions = [d for d, pos in cerebro.broker.positions.items() if pos.size > 0]
    if open_positions:
        print("\n当前活跃持仓 (未平仓):")
        print(f"{'代码':<12} {'数量':<10} {'当前价':<10} {'总市值':<12}")
        for d in open_positions:
            pos = cerebro.broker.getposition(d)
            cur_price = d.close[0]
            val = pos.size * cur_price
            print(f"{d._name:<12} {pos.size:<10} {cur_price:<10.2f} {val:,.2f}")
    else:
        print("\n当前无活跃持仓。")

    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"\n最大回撤: {dd.max.drawdown:.2f}%")
        trade_stats = strat.analyzers.trades.get_analysis()
        # 处理总交易次数：平仓交易 + 当前持仓交易
        closed_count = trade_stats.total.total
        total_trades_all = closed_count + len(open_positions)
        print(f"已平仓交易数: {closed_count}")
        print(f"当前持仓股票: {len(open_positions)}")
        print(f"总参与交易数: {total_trades_all}")
        if closed_count > 0:
            print(f"已平仓胜率: {(trade_stats.won.total / closed_count)*100:.2f}%")
    except Exception: pass
    print("="*50)

if __name__ == "__main__":
    run_full_market_backtest()
