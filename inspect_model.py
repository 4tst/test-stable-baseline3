"""
检查 PPO 模型的详细信息
展示模型包含的所有信息和结构
"""

from stable_baselines3 import PPO
import torch
import numpy as np
import os


def inspect_model(model_path="ppo_snake_v1.zip"):
    """详细检查 PPO 模型"""
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练: python snake.py train")
        return
    
    print("=" * 70)
    print("   🤖 PPO 模型详细信息检查")
    print("=" * 70)
    print()
    
    # 加载模型
    print("📦 加载模型...")
    model = PPO.load(model_path)
    print(f"✅ 模型加载成功: {model_path}")
    print()
    
    # 1. 基本信息
    print("=" * 70)
    print("   1️⃣  基本信息")
    print("=" * 70)
    print(f"算法: PPO (Proximal Policy Optimization)")
    print(f"策略类型: MlpPolicy (多层感知机)")
    print(f"观察空间: {model.observation_space}")
    print(f"动作空间: {model.action_space}")
    print()
    
    # 2. 网络结构
    print("=" * 70)
    print("   2️⃣  网络结构")
    print("=" * 70)
    print()
    print("📊 完整策略网络:")
    print(model.policy)
    print()
    
    # 3. 参数统计
    print("=" * 70)
    print("   3️⃣  参数统计")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print()
    
    # 计算存储空间
    param_size_bytes = total_params * 4  # float32
    print(f"存储空间估算:")
    print(f"  • 纯权重: {param_size_bytes / 1024:.2f} KB")
    print(f"  • 含优化器: {param_size_bytes * 4 / 1024:.2f} KB")
    print(f"  • ZIP压缩后: ~{os.path.getsize(model_path) / 1024:.2f} KB")
    print()
    
    # 4. 超参数
    print("=" * 70)
    print("   5️⃣  训练超参数")
    print("=" * 70)
    print()
    
    hyperparams = {
        'learning_rate': model.lr_schedule(1),
        'n_steps': model.n_steps,
        'batch_size': model.batch_size,
        'n_epochs': model.n_epochs,
        'gamma': model.gamma,
        'gae_lambda': model.gae_lambda,
        'clip_range': model.clip_range(1),
        'ent_coef': model.ent_coef,
        'vf_coef': model.vf_coef,
        'max_grad_norm': model.max_grad_norm,
    }
    
    for key, value in hyperparams.items():
        print(f"  {key:20s}: {value}")
    print()
    
    # 5. 测试预测
    print("=" * 70)
    print("   5️⃣  测试预测功能")
    print("=" * 70)
    print()
    
    # 创建测试观察
    test_obs = np.array([0.5, 0.5, 0.7, 0.3, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    
    print(f"测试观察向量: {test_obs}")
    print(f"  [蛇头X, 蛇头Y, 食物X, 食物Y, 前危, 左危, 右危, 食向]")
    print()
    
    # 确定性预测
    action_det, value_det = model.predict(test_obs, deterministic=True)
    print(f"确定性预测:")
    print(f"  动作: {action_det} ({['直走', '右转', '左转'][action_det]})")
    print(f"  价值: {value_det:.2f}")
    print()
    
    # 概率分布
    print(f"动作概率分布:")
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0)
        action_dist = model.policy.get_distribution(obs_tensor)
        probs = action_dist.distribution.probs[0]
        
        actions = ['直走 (0)', '右转 (1)', '左转 (2)']
        for i, (prob, name) in enumerate(zip(probs, actions)):
            prob_val = prob.item()
            bar_len = int(prob_val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {name:12s}: {prob_val:.4f} |{bar}|")
    print()
    
    # 6. 模型文件信息
    print("=" * 70)
    print("   6️⃣  模型文件信息")
    print("=" * 70)
    print()
    
    file_size = os.path.getsize(model_path)
    print(f"文件名: {model_path}")
    print(f"文件大小: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # 修改时间
    import time
    mtime = os.path.getmtime(model_path)
    print(f"修改时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    print()
    
    # 7. 总结
    print("=" * 70)
    print("   📝 总结")
    print("=" * 70)
    print()
    print("这个模型包含:")
    print("  ✅ 策略网络: 8 → 128 → 128 → 3 (决定动作)")
    print("  ✅ 价值网络: 8 → 128 → 128 → 1 (评估状态)")
    print(f"  ✅ 总参数量: {total_params:,} 个浮点数")
    print(f"  ✅ 文件大小: {file_size / 1024:.2f} KB (ZIP压缩)")
    print()
    print("工作原理:")
    print("  1. 输入 8 维观察向量")
    print("  2. 经过两层神经网络处理")
    print("  3. 输出 3 个动作的概率分布")
    print("  4. 选择概率最高的动作执行")
    print()
    print("如何使用:")
    print("  from stable_baselines3 import PPO")
    print("  model = PPO.load('ppo_snake_v1')")
    print("  action, _ = model.predict(observation)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    inspect_model()