"""
演示 render_mode 的不同用法
"""

from snake import SnakeEnv
import time


def demo_render_modes():
    """演示不同的 render_mode"""
    
    print("=" * 70)
    print("   🎨 render_mode 演示")
    print("=" * 70)
    print()
    
    # 模式 1: None (不渲染)
    print("1️⃣  模式: render_mode=None (训练模式)")
    print("-" * 70)
    env1 = SnakeEnv(render_mode=None)
    obs, _ = env1.reset()
    
    start_time = time.time()
    steps = 1000
    for _ in range(steps):
        action = env1.action_space.sample()
        obs, reward, terminated, truncated, _ = env1.step(action)
        if terminated or truncated:
            obs, _ = env1.reset()
    
    elapsed = time.time() - start_time
    fps = steps / elapsed
    
    print(f"运行 {steps} 步耗时: {elapsed:.2f} 秒")
    print(f"FPS: {fps:.0f}")
    print(f"特点: ⚡ 速度快，无图形界面，适合训练")
    print()
    
    # 模式 2: human (显示窗口)
    print("2️⃣  模式: render_mode='human' (可视化模式)")
    print("-" * 70)
    print("这种模式会打开 Pygame 窗口，实时显示游戏画面")
    print("特点: 🎮 可以看到游戏，但速度慢，适合测试和演示")
    print()
    print("使用场景:")
    print("  • 测试 AI 模型")
    print("  • 人类玩家模式")
    print("  • 录制演示视频")
    print()
    
    # 总结
    print("=" * 70)
    print("   📝 总结")
    print("=" * 70)
    print()
    print("render_mode 的作用:")
    print("  ✅ 控制是否显示图形界面")
    print("  ✅ 训练时用 None (快)")
    print("  ✅ 测试时用 'human' (可见)")
    print("  ✅ 提高灵活性和性能")
    print()
    print("在你的项目中:")
    print("  • train() 函数: SnakeEnv()           # 无渲染")
    print("  • test() 函数:  SnakeEnv(render_mode='human')  # 有渲染")
    print("  • play_human(): SnakeEnv(render_mode='human')  # 有渲染")
    print()


if __name__ == "__main__":
    demo_render_modes()