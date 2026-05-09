# 🚀 提高 AI 得分的完整优化方案

## 📊 当前问题分析

**现状**：
- AI 平均只能玩 30 步左右
- 平均得分约 5-8 分
- 容易撞墙或撞到自己

**目标**：
- 平均步数 > 100 步
- 平均得分 > 15 分
- 最高得分 > 30 分

---

## 🔧 优化方案

### 方案 1: 增加训练步数（最有效）⭐⭐⭐⭐⭐

**问题**：当前训练 1,000,000 步可能不够

**解决**：
```python
# 在 train() 函数中修改
model.learn(total_timesteps=2000000)  # 从 100万 增加到 200万
# 或甚至
model.learn(total_timesteps=5000000)  # 500万步
```

**预期效果**：
- 训练时间增加 2-5 倍
- AI 表现提升 50-100%
- 平均得分可达 15-25 分

---

### 方案 2: 调整超参数 ⭐⭐⭐⭐

#### 2.1 增加探索能力
```python
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.03,  # 从 0.02 增加到 0.03，鼓励更多探索
    ...
)
```

#### 2.2 调整学习率
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,  # 从 3e-4 降低到 1e-4，更稳定的学习
    ...
)
```

#### 2.3 增大网络
```python
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # 从 128 增加到 256
    )
)
```

---

### 方案 3: 改进奖励函数 ⭐⭐⭐⭐

**当前问题**：AI 可能还没学会长期规划

**优化建议**：

#### 3.1 增加存活奖励
```python
def step(self, action):
    # ... 现有代码 ...
    
    if new_head == self.food:
        reward = 10
        # ...
    else:
        # 新增：存活奖励（鼓励活得更久）
        reward = 0.01  # 每存活一步给小奖励
        
        # 距离引导（已有）
        distance_change = old_distance - new_distance
        if distance_change > 0:
            reward += 0.1  # 向食物靠近
        elif distance_change < 0:
            reward -= 0.2  # 远离食物
```

#### 3.2 增加转弯惩罚
```python
# 如果 AI 频繁转弯，给予额外惩罚
if action != 0:  # 不是直走
    reward -= 0.01  # 轻微惩罚不必要的转弯
```

---

### 方案 4: 改进观察空间 ⭐⭐⭐

**当前观察空间**（8维）：
```python
[蛇头X, 蛇头Y, 食物X, 食物Y, 前危, 左危, 右危, 食向]
```

**可以添加的信息**：

#### 4.1 蛇身长度
```python
obs = np.array([
    # ... 现有的 8 维 ...
    len(self.snake) / 100,  # 归一化的蛇身长度
])
```

#### 4.2 周围格子的占用情况
```python
# 检查蛇头周围 8 个格子是否有障碍物
surrounding_danger = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
            continue
        pos = (head_x + dx, head_y + dy)
        surrounding_danger.append(self._is_collision(pos))

obs = np.concatenate([existing_obs, surrounding_danger])
```

---

### 方案 5: 使用向量环境加速训练 ⭐⭐⭐⭐

**原理**：同时运行多个环境，并行收集经验

```python
from stable_baselines3.common.env_util import make_vec_env

def train():
    # 创建 4 个并行环境
    env = make_vec_env(lambda: SnakeEnv(), n_envs=4)
    
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=1000000)
```

**优势**：
- 训练速度提升 3-4 倍
- 经验更多样化
- 收敛更快

---

### 方案 6: 课程学习 ⭐⭐⭐

**思路**：从简单到复杂逐步训练

```python
# 阶段 1: 小网格训练（10x10）
env = SnakeEnv(grid_size=10)
model.learn(total_timesteps=200000)

# 阶段 2: 中等网格（15x15）
env = SnakeEnv(grid_size=15)
model.set_env(env)
model.learn(total_timesteps=300000)

# 阶段 3: 标准网格（20x20）
env = SnakeEnv(grid_size=20)
model.set_env(env)
model.learn(total_timesteps=500000)
```

---

## 🎯 推荐的实施步骤

### 第一步：立即尝试（最快见效）

1. **增加训练步数到 200 万**
2. **稍微增加探索系数**

修改 `snake.py` 中的 `train()` 函数：

```python
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,  # ← 从 0.02 增加到 0.03
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
)

model.learn(total_timesteps=2000000)  # ← 从 100万 增加到 200万
```

**预期**：训练时间翻倍，但 AI 表现显著提升。

---

### 第二步：如果还不够（进一步优化）

1. **降低学习率**：`learning_rate=1e-4`
2. **增大网络**：`net_arch=dict(pi=[256, 256], vf=[256, 256])`
3. **继续增加训练步数**：`total_timesteps=3000000`

---

### 第三步：高级优化（如果需要更高分数）

1. **使用向量环境**（4-8 个并行环境）
2. **添加更多观察特征**
3. **实施课程学习**
4. **多次训练取最优模型**

---

## 📈 预期效果对比

| 优化方案 | 训练时间 | 平均得分 | 最高得分 | 难度 |
|---------|---------|---------|---------|------|
| 当前状态 | 2-4小时 | 5-8 | 10-15 | - |
| 方案1: 增加训练步数 | 4-8小时 | 12-18 | 20-30 | ⭐ |
| 方案2: 调整超参数 | 4-8小时 | 15-22 | 25-35 | ⭐⭐ |
| 方案1+2: 组合优化 | 6-10小时 | 18-25 | 30-45 | ⭐⭐ |
| 方案5: 向量环境 | 2-3小时 | 15-20 | 25-35 | ⭐⭐⭐ |
| 全部方案 | 10-20小时 | 25-35 | 50+ | ⭐⭐⭐⭐⭐ |

---

## 💡 快速测试脚本

创建一个脚本来测试不同配置的效果：

```python
"""test_ai_performance.py - 测试 AI 性能"""
from snake import SnakeEnv
from stable_baselines3 import PPO
import numpy as np

def test_model(model_path, num_episodes=20):
    """测试模型的平均表现"""
    env = SnakeEnv()
    model = PPO.load(model_path, env=env)
    
    scores = []
    lengths = []
    
    for ep in range(num_episodes):
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
    
    print(f"\n{'='*50}")
    print(f"测试结果 ({num_episodes} 局)")
    print(f"{'='*50}")
    print(f"平均得分: {np.mean(scores):.2f}")
    print(f"最高得分: {max(scores)}")
    print(f"最低得分: {min(scores)}")
    print(f"平均步数: {np.mean(lengths):.2f}")
    print(f"最长生存: {max(lengths)} 步")
    print(f"{'='*50}\n")
    
    return np.mean(scores)

if __name__ == "__main__":
    test_model("ppo_snake_v1.zip")
```

---

## 🚀 立即行动

### 选项 A: 保守优化（推荐先试这个）

修改 `snake.py`：
```python
ent_coef=0.03,  # 增加探索
model.learn(total_timesteps=2000000)  # 增加训练
```

### 选项 B: 激进优化

修改 `snake.py`：
```python
learning_rate=1e-4,  # 降低学习率
ent_coef=0.03,       # 增加探索
policy_kwargs=dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])  # 增大网络
)
model.learn(total_timesteps=3000000)  # 大幅增加训练
```

---

## ❓ 常见问题

### Q: 为什么要训练这么久？
A: 强化学习需要大量试错才能学到好策略。就像人类学习玩游戏一样，需要练习很多次。

### Q: 有没有更快的方法？
A: 使用 GPU 可以加速 5-10 倍，或使用向量环境并行训练。

### Q: 如何知道训练是否足够？
A: 监控 `ep_rew_mean` 指标，当它趋于平稳不再上升时，说明基本收敛。

### Q: 能否中途停止并继续训练？
A: 可以！保存模型后，下次加载继续训练即可。

---

**建议**：先从**选项 A** 开始，如果效果还不够好，再尝试**选项 B**。

记住：**训练时间是换取 AI 智能的必要投资**！🎯