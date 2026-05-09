"""
贪吃蛇游戏 - 快速启动指南

运行方式:
1. python snake.py                    # 交互式选择模式
2. python snake.py --mode train       # 直接训练AI
3. python snake.py --mode test        # 直接测试AI
4. python snake.py --mode play        # 直接进入人类玩家模式

或者直接在代码中修改 __main__ 部分来设置默认模式
"""

import sys
from snake import train, test, play_human


if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train":
            train()
        elif mode == "test":
            test()
        elif mode == "play":
            play_human()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: train, test, play")
    else:
        # 默认交互模式
        print("=== Snake Game ===")
        print("请选择模式:")
        print("1. 训练AI (train)")
        print("2. AI测试 (test)")
        print("3. 人类玩家模式 (play)")

        choice = input("请输入选择 (1/2/3): ").strip()

        if choice == "1" or choice.lower() == "train":
            train()
        elif choice == "2" or choice.lower() == "test":
            test()
        elif choice == "3" or choice.lower() == "play":
            play_human()
        else:
            print("无效选择，默认进入人类玩家模式")
            play_human()
