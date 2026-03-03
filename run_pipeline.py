import subprocess
import time
import socket
import os
import sys

# ================= 配置路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(BASE_DIR, "quant_stock_env", "Scripts", "python.exe")
REDIS_EXE = os.path.join(BASE_DIR, "Redis-x64-5.0.14.1", "redis-server.exe")
REDIS_CONF = os.path.join(BASE_DIR, "Redis-x64-5.0.14.1", "redis.windows.conf")
QUESTDB_EXE = os.path.join(BASE_DIR, "questdb", "questdb-9.3.3-rt-windows-x86-64", "bin", "questdb.exe")

def check_port(port):
    """检测本地端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_service(name, port, command, cwd=None):
    """启动后台服务"""
    if check_port(port):
        print(f"[OK] {name} 已经在运行 (端口: {port})")
    else:
        print(f"[Start] 正在启动 {name}...")
        try:
            subprocess.Popen(command, shell=True, cwd=cwd, 
                             creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
            # 等待服务初始化
            for _ in range(10):
                time.sleep(1)
                if check_port(port):
                    print(f"[OK] {name} 启动成功")
                    return
            print(f"[Error] {name} 启动超时，请检查路径或日志。")
        except Exception as e:
            print(f"[Error] 无法启动 {name}: {e}")

def run_script(script_name):
    """执行 Python 脚本"""
    print(f"\n>>> 正在执行: {script_name}")
    try:
        subprocess.run([PYTHON_EXE, script_name], check=True)
        print(f"[Done] {script_name} 执行完毕")
    except subprocess.CalledProcessError:
        print(f"[Error] {script_name} 执行失败，流程中断。")
        sys.exit(1)

if __name__ == "__main__":
    print("="*50)
    print("   QuantStock 量化全流程一键启动脚本")
    print("="*50)

    # 1. 启动基础设施服务
    start_service("Redis", 6379, f'"{REDIS_EXE}" "{REDIS_CONF}"')
    start_service("QuestDB", 8812, f'"{QUESTDB_EXE}"')

    # 2. 执行量化流水线
    pipeline = [
        "alpha_evaluator.py",
        "vectorbt_backtest.py",
        # "backtrader_backtest.py"
    ]

    for script in pipeline:
        run_script(script)

    print("\n" + "="*50)
    print("[Complete] 所有评估与回测步骤已圆满完成！")
    print("="*50)
