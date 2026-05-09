# PPO 训练详解 - 工作原理与模型结构

## 📚 目录

1. [训练阶段工作流程](#训练阶段工作流程)
2. [模型包含的信息](#模型包含的信息)
3. [模型文件结构](#模型文件结构)
4. [如何查看模型内容](#如何查看模型内容)
5. [技术细节](#技术细节)

---

## 🔄 训练阶段工作流程

### 阶段 1: 初始化

```python
# 1. 创建环境
env = SnakeEnv()

# 2. 初始化 PPO 模型
model = PPO(
    "MlpPolicy",      # 使用多层感知机策略
    env,              # 游戏环境
    learning_rate=3e-4,
    n_steps=2048,     # 每次收集 2048 步经验
    batch_size=64,    # 批量大小
    n_epochs=10,      # 每批数据更新 10 次
    ...
)
```

**此时创建的组件**：
- **策略网络 (Policy Network)**: 决定采取什么动作
- **价值网络 (Value Network)**: 评估当前状态的价值
- **优化器**: 用于更新网络权重
- **经验缓冲区**: 存储交互数据

### 阶段 2: 经验收集 (Rollout)

```
for step in range(n_steps):  # 2048 步
    
    1. AI 观察当前状态 s_t
       obs = [蛇头位置, 食物位置, 危险检测, ...]
    
    2. 策略网络输出动作概率
       action_probs = policy_network(obs)
       # 例如: [直走: 0.6, 右转: 0.3, 左转: 0.1]
    
    3. 采样动作（带探索）
       action = sample(action_probs)
       # 例如: 选择 "右转"
    
    4. 执行动作，获得新状态和奖励
       new_obs, reward, done, info = env.step(action)
       # 例如: reward = 0.1 (向食物靠近)
    
    5. 记录经验 tuple
       buffer.add(s_t, action, reward, s_{t+1}, done)
```

**收集的数据**（每个时间步）：
```python
{
    'observations': [8维向量],    # 当前状态
    'actions': [整数 0/1/2],      # 采取的动作
    'rewards': [浮点数],           # 获得的奖励
    'next_observations': [8维向量], # 下一个状态
    'dones': [布尔值],             # 是否结束
    'values': [浮点数],            # 价值网络的预测
    'log_probs': [浮点数],         # 动作的对数概率
}
```

### 阶段 3: 计算优势函数 (GAE)

**目的**: 评估每个动作比平均水平好多少

```python
# 广义优势估计 (Generalized Advantage Estimation)
advantages = []
for t in reversed(range(n_steps)):
    if t is last step:
        delta = rewards[t] - values[t]
    else:
        delta = rewards[t] + gamma * values[t+1] - values[t]
    
    advantage = delta + gamma * gae_lambda * advantage
    advantages.insert(0, advantage)

# 计算回报 (Returns)
returns = advantages + values
```

**直观理解**：
- `advantage > 0`: 这个动作比预期好
- `advantage < 0`: 这个动作比预期差
- `advantage = 0`: 这个动作符合预期

### 阶段 4: 策略网络更新

**目标**: 让好的动作更可能被选择，坏的动作更少被选择

```python
for epoch in range(n_epochs):  # 10 次
    
    for batch in mini_batches:  # 分批处理
        
        # 1. 前向传播
        action_logits = policy_network(batch['observations'])
        action_log_probs = log_softmax(action_logits)
        
        # 2. 计算策略损失 (PPO Clip Loss)
        ratio = exp(action_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = clip(ratio, 1-ε, 1+ε) * advantages
        policy_loss = -min(surrogate1, surrogate2).mean()
        
        # 3. 添加熵正则化（鼓励探索）
        entropy = -sum(prob * log(prob))
        policy_loss -= ent_coef * entropy
        
        # 4. 反向传播
        policy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**PPO 的核心创新 - Clipping**:
```
ratio = π_new(a|s) / π_old(a|s)

如果 ratio 在 [0.8, 1.2] 范围内:
    正常更新
    
如果 ratio 超出范围:
    截断，防止更新过大导致策略崩溃
```

### 阶段 5: 价值网络更新

**目标**: 让价值网络更准确地预测状态价值

```python
for epoch in range(n_epochs):
    
    for batch in mini_batches:
        
        # 1. 前向传播
        predicted_values = value_network(batch['observations'])
        
        # 2. 计算价值损失 (MSE)
        value_loss = MSE(predicted_values, batch['returns'])
        
        # 3. 反向传播
        value_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 阶段 6: 重复迭代

```
总训练步数 = 1,000,000
每次收集 = 2,048 步
更新次数 = 1,000,000 / 2,048 ≈ 488 次

每次更新包含:
  - 10 个 epoch
  - 每个 epoch 处理 2048/64 = 32 个 batch
  
总更新次数 = 488 × 10 × 32 = 156,160 次梯度更新
```

---

## 📦 模型包含的信息

### 保存的模型文件 (`ppo_snake_v1.zip`) 包含：

#### 1. **策略网络权重 (Policy Network Weights)**

```
神经网络结构:
输入层 (8维) → 隐藏层1 (128神经元) → 隐藏层2 (128神经元) → 输出层 (3维)

权重矩阵:
- layer_1.weight: [128 × 8]    # 第一层权重
- layer_1.bias: [128]          # 第一层偏置
- layer_2.weight: [128 × 128]  # 第二层权重
- layer_2.bias: [128]          # 第二层偏置
- output.weight: [3 × 128]     # 输出层权重
- output.bias: [3]             # 输出层偏置

总参数数量 ≈ (128×8 + 128) + (128×128 + 128) + (3×128 + 3)
           = 1,152 + 16,512 + 387
           = 18,051 个参数
```

**作用**: 将观察空间映射到动作概率分布
```python
输入: obs = [0.5, 0.5, 0.7, 0.3, 0.0, 0.0, 0.0, 0.2]
      ↓ (经过神经网络)
输出: action_probs = [0.6, 0.3, 0.1]
      含义: 60%概率直走，30%右转，10%左转
```

#### 2. **价值网络权重 (Value Network Weights)**

```
神经网络结构:
输入层 (8维) → 隐藏层1 (128神经元) → 隐藏层2 (128神经元) → 输出层 (1维)

权重矩阵:
- layer_1.weight: [128 × 8]
- layer_1.bias: [128]
- layer_2.weight: [128 × 128]
- layer_2.bias: [128]
- output.weight: [1 × 128]
- output.bias: [1]

总参数数量 ≈ 18,049 个参数
```

**作用**: 预测当前状态的长期价值
```python
输入: obs = [0.5, 0.5, 0.7, 0.3, 0.0, 0.0, 0.0, 0.2]
      ↓ (经过神经网络)
输出: value = 15.3
      含义: 从这个状态开始，预期能获得 15.3 的累计奖励
```

#### 3. **优化器状态 (Optimizer State)**

```python
{
    'state': {
        # Adam 优化器的内部状态
        'exp_avg': {...},      # 一阶矩估计
        'exp_avg_sq': {...},   # 二阶矩估计
    },
    'param_groups': [
        {
            'lr': 3e-4,        # 学习率
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            ...
        }
    ]
}
```

**作用**: 保存优化器的内部状态，使得可以继续训练

#### 4. **训练超参数 (Hyperparameters)**

```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.02,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': {
        'net_arch': [dict(pi=[128, 128], vf=[128, 128])]
    }
}
```

#### 5. **环境信息 (Environment Info)**

```python
{
    'observation_space': Box(8,),     # 观察空间维度
    'action_space': Discrete(3),      # 动作空间维度
    'grid_size': 20,                  # 网格大小
    'block_size': 20,                 # 像素大小
}
```

#### 6. **训练统计信息 (Training Stats)**

```python
{
    'time_elapsed': 7200.5,           # 训练耗时（秒）
    'total_timesteps': 1000000,       # 总训练步数
    'n_updates': 4880,                # 更新次数
}
```

---

## 📁 模型文件结构

### ZIP 文件内部结构

```
ppo_snake_v1.zip
├── data/
│   ├── policy.pt              # 策略网络权重 (PyTorch)
│   ├── policy.optimizer.pt    # 策略优化器状态
│   ├── value.pt               # 价值网络权重
│   ├── value.optimizer.pt     # 价值优化器状态
│   └── training_params.json   # 训练参数
├── pytorch_variables.pth      # PyTorch 变量
└── system_info.json           # 系统信息
```

### 实际文件大小

```
ppo_snake_v1.zip: ~500 KB - 2 MB

分解:
- 策略网络权重: ~200 KB
- 价值网络权重: ~200 KB
- 优化器状态: ~400 KB
- 其他元数据: ~50 KB
```

---

## 🔍 如何查看模型内容

### 方法 1: 使用 Stable-Baselines3 API

```python
from stable_baselines3 import PPO

# 加载模型
model = PPO.load("ppo_snake_v1")

# 查看策略网络
print(model.policy)
# 输出:
# ActorCriticPolicy(
#   (features_extractor): FlattenExtractor(...)
#   (actor): Sequential(
#     (0): Linear(in_features=8, out_features=128)
#     (1): Tanh()
#     (2): Linear(in_features=128, out_features=128)
#     (3): Tanh()
#   )
#   (critic): Sequential(...)
# )

# 查看网络权重
for name, param in model.policy.named_parameters():
    print(f"{name}: shape={param.shape}, mean={param.mean():.4f}")

# 测试预测
import numpy as np
obs = np.array([0.5, 0.5, 0.7, 0.3, 0.0, 0.0, 0.0, 0.2])
action, _ = model.predict(obs, deterministic=True)
print(f"预测动作: {action}")
```

### 方法 2: 直接查看 ZIP 文件

```python
import zipfile
import json

# 查看 ZIP 内容
with zipfile.ZipFile("ppo_snake_v1.zip", 'r') as zip_ref:
    print("文件列表:")
    for name in zip_ref.namelist():
        print(f"  {name}")
    
    # 读取训练参数
    with zip_ref.open('data/training_params.json') as f:
        params = json.load(f)
        print("\n训练参数:")
        print(json.dumps(params, indent=2))
```

### 方法 3: 导出为 ONNX 格式（跨平台）

```python
import torch
from stable_baselines3 import PPO

model = PPO.load("ppo_snake_v1")

# 导出策略网络
dummy_input = torch.randn(1, 8)
torch.onnx.export(
    model.policy.actor,
    dummy_input,
    "policy.onnx",
    input_names=['observation'],
    output_names=['action_logits']
)

print("✅ 已导出为 ONNX 格式")
```

### 方法 4: 可视化网络结构

```python
import torch
from torchviz import make_dot
from stable_baselines3 import PPO

model = PPO.load("ppo_snake_v1")

# 创建虚拟输入
obs = torch.randn(1, 8)

# 前向传播
logits = model.policy.actor(obs)

# 可视化
dot = make_dot(logits, params=dict(model.policy.actor.named_parameters()))
dot.render("policy_network", format="png")
```

---

## ⚙️ 技术细节

### 1. 神经网络的数学表示

**策略网络**:
```
输入: x ∈ R^8

第1层: h1 = tanh(W1·x + b1)    # W1 ∈ R^(128×8), b1 ∈ R^128
第2层: h2 = tanh(W2·h1 + b2)   # W2 ∈ R^(128×128), b2 ∈ R^128
输出层: logits = W3·h2 + b3     # W3 ∈ R^(3×128), b3 ∈ R^3

动作概率: π(a|x) = softmax(logits)
```

**价值网络**:
```
输入: x ∈ R^8

第1层: h1 = tanh(V1·x + c1)    # V1 ∈ R^(128×8), c1 ∈ R^128
第2层: h2 = tanh(V2·h1 + c2)   # V2 ∈ R^(128×128), c2 ∈ R^128
输出: V(x) = V3·h2 + c3         # V3 ∈ R^(1×128), c3 ∈ R^1

状态价值: V(x) ∈ R
```

### 2. 参数数量计算

```
策略网络:
- 第1层: 128 × 8 + 128 = 1,152
- 第2层: 128 × 128 + 128 = 16,512
- 输出层: 3 × 128 + 3 = 387
总计: 18,051 个参数

价值网络:
- 第1层: 128 × 8 + 128 = 1,152
- 第2层: 128 × 128 + 128 = 16,512
- 输出层: 1 × 128 + 1 = 129
总计: 17,793 个参数

总参数量: 35,844 个

存储空间:
- float32: 4 bytes/参数
- 总大小: 35,844 × 4 = 143,376 bytes ≈ 140 KB
- 加上优化器状态（约3倍）: ~500 KB
```

### 3. 训练过程中的内存占用

```
经验缓冲区 (2048 步):
- observations: 2048 × 8 × 4 = 64 KB
- actions: 2048 × 4 = 8 KB
- rewards: 2048 × 4 = 8 KB
- values: 2048 × 4 = 8 KB
- log_probs: 2048 × 4 = 8 KB
小计: ~100 KB

网络权重:
- 策略网络: ~140 KB
- 价值网络: ~140 KB
- 优化器状态: ~800 KB
小计: ~1 MB

总内存占用: ~2-5 MB (很小！)
```

### 4. 推理速度

```
单次预测:
- 前向传播: 8 → 128 → 128 → 3
- 计算量: 约 50,000 次浮点运算
- CPU 时间: ~0.1-0.5 ms
- GPU 时间: ~0.01-0.05 ms

每秒可预测: 
- CPU: 2,000-10,000 次
- GPU: 20,000-100,000 次
```

---

## 💡 关键概念总结

### 训练做了什么？

1. **收集经验**: AI 与环境交互，记录状态-动作-奖励序列
2. **评估动作**: 使用 GAE 计算每个动作的优势值
3. **更新策略**: 调整策略网络，让好动作更可能，坏动作更不可能
4. **更新价值**: 调整价值网络，使其更准确预测状态价值
5. **重复迭代**: 上述过程重复数十万次

### 模型包含什么？

1. **策略网络权重**: 决定采取什么动作（~18K 参数）
2. **价值网络权重**: 评估状态价值（~18K 参数）
3. **优化器状态**: 支持继续训练
4. **超参数**: 训练配置
5. **元数据**: 环境信息、训练统计等

### 如何使用模型？

```python
# 加载
model = PPO.load("ppo_snake_v1")

# 预测
action, _ = model.predict(observation)

# 本质是:
# 1. 将 observation 输入策略网络
# 2. 网络输出动作概率分布
# 3. 选择概率最高的动作（deterministic=True）
# 4. 返回动作编号
```

---

## 🎯 实际示例

### 查看你的模型

运行以下脚本查看模型详情：

```python
"""inspect_model.py - 检查 PPO 模型"""
from stable_baselines3 import PPO
import torch

# 加载模型
model = PPO.load("ppo_snake_v1")

print("=" * 60)
print("   PPO 模型详细信息")
print("=" * 60)

# 1. 网络结构
print("\n📊 网络结构:")
print(model.policy)

# 2. 参数统计
print("\n🔢 参数统计:")
total_params = sum(p.numel() for p in model.policy.parameters())
trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
print(f"总参数数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# 3. 权重统计
print("\n📈 权重统计:")
for name, param in model.policy.named_parameters():
    if 'weight' in name:
        print(f"{name:30s}: shape={str(param.shape):20s} "
              f"mean={param.mean():8.4f} std={param.std():8.4f}")

# 4. 测试预测
print("\n🧪 测试预测:")
import numpy as np
test_obs = np.array([0.5, 0.5, 0.7, 0.3, 0.0, 0.0, 0.0, 0.2])
action, value = model.predict(test_obs, deterministic=True)
print(f"输入观察: {test_obs}")
print(f"预测动作: {action}")
print(f"状态价值: {value:.2f}")

# 5. 动作概率
print("\n🎲 动作概率:")
with torch.no_grad():
    obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0)
    action_dist = model.policy.get_distribution(obs_tensor)
    probs = action_dist.distribution.probs[0]
    actions = ['直走', '右转', '左转']
    for i, (prob, name) in enumerate(zip(probs, actions)):
        bar = "█" * int(prob.item() * 30)
        print(f"  {name:4s}: {prob.item():.3f} {bar}")

print("\n" + "=" * 60)
```

---

**版本**: v1.0  
**更新日期**: 2026-05-09  
**适用模型**: PPO (Stable-Baselines3)