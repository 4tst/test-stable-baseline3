"""
测试 AI 模型的性能表现
运行多局游戏，统计平均得分和最高得分
"""

from snake import SnakeEnv
from stable_baselines3 import PPO
import numpy as np
import os


def test_model(model_path="ppo_snake_v1.zip", num_episodes=20):
    """
    测试模型的平均表现
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试局数
    """
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练: python snake.py train")
        return
    
    print("=" * 70)
    print(f"   🧪 AI 性能测试 - {num_episodes} 局")
    print("=" * 70)
    print()
    
    # 加载模型
    env = SnakeEnv()
    model = PPO.load(model_path, env=env)
    print(f"✅ 模型加载成功: {model_path}")
    print()
    
    # 测试多局
    scores = []
    lengths = []
    foods_eaten = []
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        score = len(env.snake) - 3
        scores.append(score)
        lengths.append(steps)
        foods_eaten.append(score)
        
        # 显示进度
        if ep % 5 == 0 or ep == num_episodes:
            print(f"第 {ep:3d} 局: 得分={score:3d}, 步数={steps:4d}")
    
    # 统计分析
    print()
    print("=" * 70)
    print("   📊 测试结果统计")
    print("=" * 70)
    print()
    print(f"测试局数: {num_episodes}")
    print()
    print("得分统计:")
    print(f"  • 平均得分: {np.mean(scores):.2f}")
    print(f"  • 最高得分: {max(scores)}")
    print(f"  • 最低得分: {min(scores)}")
    print(f"  • 标准差:   {np.std(scores):.2f}")
    print()
    print("步数统计:")
    print(f"  • 平均步数: {np.mean(lengths):.2f}")
    print(f"  • 最长生存: {max(lengths)} 步")
    print(f"  • 最短生存: {min(lengths)} 步")
    print()
    print("食物统计:")
    print(f"  • 平均吃到: {np.mean(foods_eaten):.2f} 个")
    print(f"  • 最多吃到: {max(foods_eaten)} 个")
    print()
    
    # 评价
    avg_score = np.mean(scores)
    max_score = max(scores)
    
    print("=" * 70)
    print("   💡 性能评价")
    print("=" * 70)
    print()
    
    if avg_score < 5:
        print("⚠️  表现较差 - 需要更多训练或调整参数")
        print("   建议:")
        print("   • 增加训练步数到 200-300 万")
        print("   • 增加探索系数 ent_coef")
        print("   • 检查奖励函数是否合理")
    elif avg_score < 10:
        print("🔶 表现一般 - 有改进空间")
        print("   建议:")
        print("   • 继续训练到 200 万步以上")
        print("   • 尝试降低学习率")
        print("   • 增大网络容量")
    elif avg_score < 15:
        print("✅ 表现良好 - 基本可用")
        print("   建议:")
        print("   • 可以继续训练提升")
        print("   • 尝试不同的超参数组合")
    elif avg_score < 25:
        print("🌟 表现优秀 - AI 很聪明！")
        print("   建议:")
        print("   • 可以尝试更高难度的挑战")
        print("   • 考虑使用向量环境加速")
    else:
        print("🏆 表现卓越 - AI 大师级水平！")
        print("   建议:")
        print("   • 已经非常厉害了")
        print("   • 可以录制演示视频分享")
    
    print()
    print("=" * 70)
    
    # 返回统计数据
    return {
        'avg_score': np.mean(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'avg_length': np.mean(lengths),
        'max_length': max(lengths),
    }


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        num_eps = int(sys.argv[1])
    else:
        num_eps = 20
    
    test_model(num_episodes=num_eps)