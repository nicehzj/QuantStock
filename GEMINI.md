# GEMINI.md - QuantStock 项目上下文

本文档为 Gemini CLI 提供项目特定的背景信息、架构说明及运行指南。

## 🚀 项目概览 (Project Overview)
**QuantStock** 是一个高性能、模块化的量化交易分析工作流系统。它旨在打通从数据抓取、存储、因子挖掘、因子评估到回测验证的全流程，并具备向高频交易扩展的能力。

### 核心技术栈
- **数据源**: Baostock (A 股日线、指数数据)
- **存储层**: 
  - **Parquet 数据湖**: 本地文件存储，一股票一文件，Snappy 压缩，极致的离线读取速度。
  - **QuestDB**: 高性能时序数据库，用于实战化数据持久化与时序聚合查询。
  - **Redis**: 内存数据库，用于缓存、任务分发及交易信号实时发布。
- **计算层**: **DuckDB** (嵌入式 OLAP 引擎)，利用 SQL 窗口函数进行毫秒级因子计算。
- **评估层**: **Alphalens**，用于因子的 IC/IR 分析及收益预测性验证。
- **回测层**: 
  - **Vectorbt**: 向量化回测，用于大规模参数寻优与初步筛选。
  - **Backtrader**: 事件驱动回测，用于模拟 A 股真实交易规则（T+1、佣金、印花税）。

## 🛠️ 运行与构建 (Building and Running)

### 环境准备
项目使用 `uv` 管理虚拟环境。
- **环境目录**: `quant_stock_env/`
- **关键依赖**: `baostock`, `pandas`, `pyarrow` (Parquet 引擎), `tqdm`.
- **安装依赖**: `uv pip install baostock pandas pyarrow tqdm`

### 核心操作流程
1. **数据抓取与同步 (`data_manager.py`)**:
   - **逻辑**: 自动识别交易日（20点后取当天，否则取前一交易日）。
   - **防幸存者偏差**: 遍历 2006 年至今每年的年初股票池并取并集，包含已退市股票。
   - **多进程加速**: 使用 `ProcessPoolExecutor` 并行下载，子进程独立维护登录 Session。
   - **本地缓存**: 股票池列表（分年及全局）缓存于 `data/stock_pools/`。

2. **数据库初始化与同步 (`questdb_manager.py`)**:
   - 初始化 QuestDB 表结构，并将本地 Parquet 数据同步至 QuestDB。

3. **因子计算与评估 (`alpha_evaluator.py`)**:
   - 计算因子、进行 Alphalens 评估，并将有效信号推送到 Redis。

4. **策略回测**:
   - **快速回测**: `python vectorbt_backtest.py`
   - **精细回测**: `python backtrader_backtest.py`

## 🏗️ 架构设计与约定 (Development Conventions)

### 模块职责
- `data_manager.py`: 负责与 Baostock 交互，处理多进程并发下载与增量更新。
- `db_connector.py`: 统一连接器，所有对 Redis, QuestDB, DuckDB 的访问必须通过此类。
- `factor_engine.py`: 因子计算核心，支持切换 `parquet` 或 `questdb` 数据源。

### 数据规范
- **复权方式**: 统一使用**后复权** (`adjustflag='1'`) 以保证历史数据的连续性和增量更新的兼容性。
- **存储格式**: 
  - 本地存储采用 `Snappy` 压缩的 Parquet，路径为 `data/daily_k/` (个股) 和 `data/index_k/` (指数)。
  - 股票池缓存路径为 `data/stock_pools/`。
  - 增量更新元数据存储于 `data/sync_metadata.json`。

### 开发注意事项
- **并发策略**: Baostock 不支持线程并发，必须使用**多进程** (`ProcessPoolExecutor`) 且每个子进程独立执行 `bs.login()`。
- **性能优化**: 核心计算优先使用 DuckDB/QuestDB 的 SQL 窗口函数。
- **数据对齐**: 原始 Parquet 不做填充，对齐逻辑下沉至分析层（如 Pandas Pivot 或 SQL Join）。

## 📝 TODO / 待办事项
- [x] 实现防幸存者偏差的全局股票池抓取。
- [x] 优化多进程同步效率与登录 Session 管理。
- [x] 实现样本外测试 (Out-of-Sample Testing) 机制 (IS: 2006-2018, OOS: 2019-Present)。
- [ ] 实现基于滚动窗口的因子有效性验证与动态权重分配 (Walk-Forward Analysis)。
- [ ] 接入分钟级/Tick 级实时行情数据。
- [ ] 实现基于 Redis Pub/Sub 的实时模拟交易模块。
- [ ] 增加更多基础 Alpha 因子库。
