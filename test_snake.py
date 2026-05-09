"""
贪吃蛇游戏自动化测试脚本
用于验证游戏环境是否正常工作
"""
from snake import SnakeEnv
import numpy as np


def test_env():
    """测试环境基本功能"""
    print("=== 开始测试贪吃蛇环境 ===\n")
    
    # 创建环境
    env = SnakeEnv()
    print("✓ 环境创建成功")
    
    # 重置环境
    obs, info = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  - 观察空间形状: {obs.shape}")
    print(f"  - 初始蛇长度: {len(env.snake)}")
    print(f"  - 初始方向: {env.direction}")
    
    # 测试随机动作
    print("\n--- 测试随机动作 ---")
    for i in range(20):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"第 {i+1} 步: 游戏结束!")
            print(f"  - 最终蛇长度: {len(env.snake)}")
            print(f"  - 最终奖励: {reward:.1f}")
            break
        else:
            if (i + 1) % 5 == 0:
                print(f"第 {i+1} 步: 动作={action}, 奖励={reward:.1f}, 蛇长度={len(env.snake)}")
    
    print("\n✓ 环境测试通过!")
    return True


def test_observation_space():
    """测试观察空间的正确性"""
    print("\n=== 测试观察空间 ===\n")
    
    env = SnakeEnv()
    obs, _ = env.reset()
    
    # 检查观察空间维度
    assert obs.shape == (8,), f"观察空间维度错误: {obs.shape}"
    print(f"✓ 观察空间维度正确: {obs.shape}")
    
    # 检查观察值范围
    assert obs.min() >= 0 and obs.max() <= 1, "观察值超出 [0, 1] 范围"
    print(f"✓ 观察值范围正确: [{obs.min():.3f}, {obs.max():.3f}]")
    
    print("\n✓ 观察空间测试通过!")
    return True


def test_collision_detection():
    """测试碰撞检测"""
    print("\n=== 测试碰撞检测 ===\n")
    
    env = SnakeEnv()
    env.reset()  # 先重置环境以初始化 snake 属性
    
    # 测试撞墙
    collision_wall = env._is_collision((-1, 0))
    assert collision_wall == 1.0, "撞墙检测失败"
    print("✓ 撞墙检测正常")
    
    # 测试安全位置
    collision_safe = env._is_collision((5, 5))
    assert collision_safe == 0.0, "安全位置检测失败"
    print("✓ 安全位置检测正常")
    
    print("\n✓ 碰撞检测测试通过!")
    return True


if __name__ == "__main__":
    try:
        test_env()
        test_observation_space()
        test_collision_detection()
        print("\n" + "="*50)
        print("🎉 所有测试通过！贪吃蛇游戏可以正常运行！")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
