"""
贪吃蛇 AI - 快速启动脚本

用法:
    python run_ai.py              # 默认测试模式
    python run_ai.py train        # 训练模式
    python run_ai.py test         # 测试模式
"""

import sys
import os

# 确保可以导入 snake 模块
sys.path.insert(0, os.path.dirname(__file__))

from snake import train, test


def main():
    print("🐍 贪吃蛇 AI 快速启动\n")
    
    # 确定运行模式
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        # 检查模型是否存在
        if os.path.exists("ppo_snake_v1.zip"):
            print("✅ 发现已训练的模型: ppo_snake_v1.zip")
            choice = input("是否测试 AI？(y/n): ").strip().lower()
            mode = "test" if choice == "y" else "train"
        else:
            print("⚠️  未找到训练好的模型")
            choice = input("是否开始训练？(y/n): ").strip().lower()
            mode = "train" if choice == "y" else "exit"
    
    # 执行相应模式
    if mode == "train":
        print("\n🎓 启动训练模式...\n")
        train()
    elif mode == "test":
        print("\n🤖 启动测试模式...\n")
        test()
    else:
        print("退出程序")
        return
    
    print("\n✅ 程序执行完毕！")


if __name__ == "__main__":
    main()