"""
测试优化后的观察空间
验证 AI 能否获得正确的信息
"""

from snake import SnakeEnv
import numpy as np


def test_observation():
    """测试观察空间的各个维度"""
    print("=" * 60)
    print("   🧪 测试优化后的观察空间")
    print("=" * 60)
    print()
    
    env = SnakeEnv()
    obs, _ = env.reset()
    
    print(f"✅ 环境初始化成功")
    print(f"✅ 观察空间形状: {obs.shape}")
    print()
    
    # 显示每个维度的含义和值
    feature_names = [
        "蛇头X (归一化)",
        "蛇头Y (归一化)",
        "食物X (归一化)",
        "食物Y (归一化)",
        "前方危险",
        "左方危险",
        "右方危险",
        "食物方向 (+=前,-=后)"
    ]
    
    print("观察空间详情:")
    print("-" * 60)
    for i, (val, name) in enumerate(zip(obs, feature_names)):
        status = "⚠️ 危险!" if val > 0.5 and i >= 4 and i <= 6 else ""
        print(f"  {i+1}. {name:20s}: {val:7.3f}  {status}")
    print("-" * 60)
    print()
    
    # 测试不同情况
    print("测试场景:")
    print()
    
    # 场景1: 检查左右危险检测
    print("1️⃣  测试左/右危险检测:")
    env.snake = [(10, 10), (9, 10), (8, 10)]
    env.direction = (1, 0)  # 向右
    obs = env._get_obs()
    print(f"   当前方向: 向右 →")
    print(f"   左方危险: {obs[5]:.3f} (应该是 0，左边安全)")
    print(f"   右方危险: {obs[6]:.3f} (应该是 0，右边安全)")
    print()
    
    # 场景2: 靠近墙壁
    print("2️⃣  测试靠近墙壁:")
    env.snake = [(19, 10), (18, 10), (17, 10)]
    env.direction = (1, 0)  # 向右（即将撞墙）
    obs = env._get_obs()
    print(f"   当前位置: (19, 10)，方向: 向右 →")
    print(f"   前方危险: {obs[4]:.3f} (应该是 1.0，前方是墙!)")
    print()
    
    # 场景3: 食物方向
    print("3️⃣  测试食物方向:")
    env.snake = [(10, 10), (9, 10), (8, 10)]
    env.direction = (1, 0)  # 向右
    env.food = (15, 10)  # 食物在右边
    obs = env._get_obs()
    food_direction = obs[7]
    print(f"   蛇头: (10, 10)，方向: 向右 →")
    print(f"   食物: (15, 10)")
    print(f"   食物方向值: {food_direction:.3f}")
    if food_direction > 0:
        print(f"   ✅ 正值表示食物在前方")
    elif food_direction < 0:
        print(f"   ❌ 负值表示食物在后方")
    else:
        print(f"   ⚠️ 零值表示食物在侧面")
    print()
    
    print("=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)
    print()
    print("💡 关键改进:")
    print("  • 第6维: 左方危险检测（之前固定为0）")
    print("  • 第7维: 右方危险检测（之前固定为0）")
    print("  • 第8维: 食物方向信息（帮助AI决定转向）")
    print()
    print("🚀 现在可以开始重新训练 AI 了!")
    print("   python snake.py train")
    print()


if __name__ == "__main__":
    test_observation()