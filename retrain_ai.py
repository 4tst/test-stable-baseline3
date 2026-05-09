"""
重新训练 AI - 解决转圈问题

这个脚本会：
1. 删除旧的模型文件（如果存在）
2. 开始新的训练
3. 使用优化后的观察空间和奖励函数
"""

import os
import sys


def main():
    print("=" * 60)
    print("   🔄 重新训练 AI - 解决转圈问题")
    print("=" * 60)
    print()
    
    # 检查并删除旧模型
    model_file = "ppo_snake_v1.zip"
    if os.path.exists(model_file):
        print(f"⚠️  发现旧模型文件: {model_file}")
        print("由于观察空间和奖励函数已优化，旧模型不再适用。")
        print()
        
        choice = input("是否删除旧模型并开始新训练？(y/n): ").strip().lower()
        if choice == 'y':
            os.remove(model_file)
            print(f"✅ 已删除旧模型: {model_file}")
        else:
            print("❌ 取消操作")
            print("提示：建议删除旧模型以获得最佳效果")
            return
    
    print()
    print("=" * 60)
    print("   🎓 开始新的训练")
    print("=" * 60)
    print()
    print("优化内容:")
    print("  ✅ 观察空间: 新增左/右危险检测 + 食物方向信息")
    print("  ✅ 奖励函数: 添加距离引导，防止转圈")
    print("  ✅ 训练参数: 增加探索能力和网络容量")
    print()
    print("预期效果:")
    print("  • AI 主动向食物移动")
    print("  • AI 避开障碍物")
    print("  • 平均得分: 8-15+")
    print()
    print("训练时间: 2-4 小时 (CPU) / 30-60 分钟 (GPU)")
    print()
    
    choice = input("是否开始训练？(y/n): ").strip().lower()
    if choice != 'y':
        print("已取消")
        return
    
    print()
    print("🚀 启动训练...\n")
    
    # 导入并运行训练
    from snake import train
    train()
    
    print()
    print("=" * 60)
    print("   ✅ 训练完成！")
    print("=" * 60)
    print()
    print("下一步:")
    print("  • 测试 AI: python snake.py test")
    print("  • 查看日志: tensorboard --logdir=./logs/")
    print()


if __name__ == "__main__":
    main()