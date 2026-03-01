import backtrader as bt
import pandas as pd
import os
from datetime import datetime

class AShareCommission(bt.CommInfoBase):
    """
    A 股专用佣金与印花税模型
    买入：仅佣金 (默认万三)
    卖出：佣金 (默认万三) + 印花税 (固定千一)
    """
    params = (
        ('stamp_duty', 0.001),      # 卖出印花税 0.1%
        ('commission', 0.0003),     # 交易佣金 0.03%
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC), # 按百分比计费
    )

    def _getcommission(self, size, price, pseudoexec):
        """计算手续费逻辑"""
        if size > 0: # 买入
            return size * price * self.p.commission
        elif size < 0: # 卖出
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        return 0

class ChinaMomentumStrategy(bt.Strategy):
    """
    Backtrader 策略类：实现 A 股精细化模拟
    包含 T+1 持仓限制及基础的调仓逻辑。
    """
    params = (
        ('top_n', 5),  # 每次调仓持有的标的数量
    )

    def __init__(self):
        # 记录每只股票的买入日期，用于实现 T+1
        self.buy_dates = {}
        # 打印日志开关
        self.log_enabled = True

    def log(self, txt, dt=None):
        if self.log_enabled:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def next(self):
        """主循环逻辑"""
        today = self.datas[0].datetime.date(0)
        
        # 1. 检查已持仓股票是否满足 T+1 卖出条件
        # A 股规定：今日买入，最早明日卖出
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                buy_date = self.buy_dates.get(data._name)
                if buy_date and today > buy_date:
                    # 模拟卖出信号（此处仅为演示：如下跌则卖出）
                    if data.close[0] < data.close[-1]:
                        self.log(f'CLOSE CREATE [卖出] {data._name}, 价格: {data.close[0]:.2f}')
                        self.sell(data=data, size=pos.size)

        # 2. 买入逻辑（示例：简单买入未持仓的股票）
        # 在实战中，此处会读取 Redis 信号或 factor_engine 计算的结果
        for data in self.datas:
            if not self.getposition(data):
                # 如果资金充足且满足买入条件
                if self.broker.get_cash() > data.close[0] * 100:
                    self.log(f'BUY CREATE [买入] {data._name}, 价格: {data.close[0]:.2f}')
                    self.buy(data=data, size=100)
                    self.buy_dates[data._name] = today

    def notify_order(self, order):
        """订单状态回调"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.data._name} 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, {order.data._name} 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}')

def run_fine_backtest(codes, start_date='2023-01-01'):
    """
    启动 Backtrader 精细回测引擎
    """
    cerebro = bt.Cerebro()
    
    print(f">>> 正在为 {len(codes)} 只股票准备精细回测数据...")
    
    # 1. 添加数据源
    for code in codes:
        file_path = f"D:/MyCode/QuantStock/data/daily_k/{code}.parquet"
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 创建 Backtrader 数据馈送
            data = bt.feeds.PandasData(
                dataname=df,
                fromdate=pd.to_datetime(start_date),
                plot=False
            )
            cerebro.adddata(data, name=code)

    # 2. 注入策略与 A 股佣金模型
    cerebro.addstrategy(ChinaMomentumStrategy)
    cerebro.broker.setcash(200000.0)
    cerebro.broker.addcommissioninfo(AShareCommission())
    
    # 3. 运行回测
    print(f'>>> 初始账户资产: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'>>> 最终账户资产: {cerebro.broker.getvalue():.2f}')

if __name__ == "__main__":
    # 选取几只具有代表性的蓝筹股进行回测流程测试
    test_list = ["sh.600519", "sz.000001", "sh.601318", "sz.000858"]
    try:
        run_fine_backtest(test_list)
    except Exception as e:
        print(f"❌ Backtrader 运行失败: {e}")
        print("💡 提示: 确保已同步上述股票的 Parquet 数据。")
