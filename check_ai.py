"""
AI 系统验证脚本
检查所有必需的组件是否正常
"""

def check_ai_system():
    """验证 AI 系统的所有组件"""
    print("=" * 60)
    print("   🤖 贪吃蛇 AI 系统验证")
    print("=" * 60)
    print()
    
    # 1. 检查依赖库
    print("1️⃣  检查依赖库...")
    try:
        import gymnasium
        print(f"   ✅ gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"   ❌ gymnasium 未安装: {e}")
        return False
    
    try:
        import pygame
        print(f"   ✅ pygame {pygame.version.ver}")
    except ImportError as e:
        print(f"   ❌ pygame 未安装: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"   ✅ stable-baselines3 {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"   ❌ stable-baselines3 未安装: {e}")
        return False
    
    try:
        import numpy
        print(f"   ✅ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"   ❌ numpy 未安装: {e}")
        return False
    
    print()
    
    # 2. 检查环境
    print("2️⃣  检查游戏环境...")
    try:
        from snake import SnakeEnv
        env = SnakeEnv()
        obs, _ = env.reset()
        print(f"   ✅ 环境创建成功")
        print(f"   ✅ 观察空间: {obs.shape}")
        print(f"   ✅ 动作空间: {env.action_space}")
    except Exception as e:
        print(f"   ❌ 环境初始化失败: {e}")
        return False
    
    print()
    
    # 3. 检查模型
    print("3️⃣  检查 PPO 模型...")
    try:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=0)
        print(f"   ✅ PPO 模型初始化成功")
        print(f"   ✅ 策略网络: MlpPolicy")
    except Exception as e:
        print(f"   ❌ 模型初始化失败: {e}")
        return False
    
    print()
    
    # 4. 检查训练函数
    print("4️⃣  检查训练功能...")
    try:
        from snake import train, test, play_human
        print(f"   ✅ train() 函数可用")
        print(f"   ✅ test() 函数可用")
        print(f"   ✅ play_human() 函数可用")
    except Exception as e:
        print(f"   ❌ 函数导入失败: {e}")
        return False
    
    print()
    
    # 5. 检查模型文件
    print("5️⃣  检查模型文件...")
    import os
    if os.path.exists("ppo_snake_v1.zip"):
        print(f"   ✅ 发现已训练的模型: ppo_snake_v1.zip")
        print(f"   💡 可以直接运行 'python snake.py test' 测试 AI")
    else:
        print(f"   ⚠️  未找到训练好的模型")
        print(f"   💡 请先运行 'python snake.py train' 进行训练")
    
    print()
    print("=" * 60)
    print("   ✅ 所有检查通过！AI 系统已就绪")
    print("=" * 60)
    print()
    print("📝 下一步:")
    print("   • 训练 AI: python snake.py train")
    print("   • 测试 AI: python snake.py test")
    print("   • 人类玩家: python snake.py play")
    print()
    
    return True


if __name__ == "__main__":
    success = check_ai_system()
    if not success:
        print("\n❌ 验证失败，请检查上述错误")
        exit(1)