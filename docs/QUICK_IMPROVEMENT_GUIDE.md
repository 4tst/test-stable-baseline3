# 🎯 提高 AI 得分 - 快速行动指南

## ⚡ 立即执行的优化（5 分钟）

### 已完成的修改

✅ **已应用保守优化方案**：
1. 探索系数：0.02 → **0.03**（增加 50%）
2. 训练步数：1,000,000 → **2,000,000**（翻倍）

### 下一步操作

#### 步骤 1: 删除旧模型（如果需要）
```bash
# Windows
del ppo_snake_v1.zip

# Linux/Mac  
rm ppo_snake_v1.zip
```

#### 步骤 2: 开始新训练
```bash
python snake.py train
```

**预期训练时间**：4-8 小时（CPU）

#### 步骤 3: 测试新模型
```bash
python test_ai_performance.py
```

---

## 📊 预期改进效果

| 指标 | 优化前 | 优化后（预期） | 提升 |
|------|--------|--------------|------|
| 平均得分 | 5-8 | **12-18** | +100% |
| 最高得分 | 10-15 | **20-30** | +100% |
| 平均步数 | 30-50 | **80-120** | +150% |
| 最长生存 | 50-80 | **150-200** | +200% |

---

## 🔍 如何监控训练进度

### 方法 1: 观察控制台输出

训练时会显示：
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 120         |  ← 平均步数（应该上升）
|    ep_rew_mean          | 25.3        |  ← 平均奖励（应该上升）
| time/                   |             |
|    total_timesteps      | 100000      |  ← 已训练步数
-----------------------------------------
```

**关键指标**：
- `ep_len_mean` > 100 ✅
- `ep_rew_mean` > 20 ✅
- 这两个值应该持续上升

### 方法 2: 使用 TensorBoard

```bash
tensorboard --logdir=./logs/
# 浏览器访问 http://localhost:6006
```

查看 `ep_rew_mean` 曲线是否持续上升。

---

## 💡 如果效果还不够好

### 进阶优化方案

#### 方案 A: 继续增加训练步数
```python
model.learn(total_timesteps=3000000)  # 300万步
```

#### 方案 B: 降低学习率（更稳定）
```python
learning_rate=1e-4,  # 从 3e-4 降到 1e-4
```

#### 方案 C: 增大网络容量
```python
policy_kwargs=dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])  # 从 128 到 256
)
```

#### 方案 D: 组合优化（推荐）
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,      # 更低的学习率
    ent_coef=0.03,           # 更高的探索
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # 更大的网络
    )
)
model.learn(total_timesteps=3000000)  # 更多训练
```

---

## 🎮 测试不同配置的脚本

创建多个模型进行比较：

```bash
# 配置 1: 当前优化
python snake.py train
mv ppo_snake_v1.zip model_config1.zip

# 配置 2: 更激进
# （修改代码后）
python snake.py train
mv ppo_snake_v1.zip model_config2.zip

# 比较两个模型
python -c "from test_ai_performance import test_model; test_model('model_config1.zip'); test_model('model_config2.zip')"
```

---

## 📈 性能基准参考

根据经验和文献，贪吃蛇 AI 的典型表现：

| 训练水平 | 平均得分 | 最高得分 | 训练步数 |
|---------|---------|---------|---------|
| 初学者 | 3-5 | 8-10 | 100k |
| 入门 | 8-12 | 15-20 | 500k |
| 熟练 | 15-20 | 25-35 | 1M |
| 优秀 | 20-30 | 40-50 | 2-3M |
| 专家 | 30-40 | 60-80 | 5M+ |
| 大师 | 40+ | 100+ | 10M+ |

**你的目标**：达到"优秀"级别（平均 20-30 分）

---

## ❓ 常见问题

### Q1: 为什么要训练这么久？
**A**: 强化学习需要大量试错。AI 要通过死亡很多次才能学会避免危险。就像人类学习玩游戏一样，需要练习。

### Q2: 能否加速训练？
**A**: 
- 使用 GPU：快 5-10 倍
- 向量环境：快 3-4 倍
- 减少日志输出：快 10-20%

### Q3: 如何知道训练是否足够？
**A**: 当 `ep_rew_mean` 不再明显上升时（趋于平稳），说明基本收敛。

### Q4: 可以中途停止吗？
**A**: 可以！按 Ctrl+C 停止，模型会保存。下次可以继续训练。

### Q5: 为什么 AI 还是容易撞墙？
**A**: 
- 可能训练还不够
- 可能需要更多探索
- 可能需要调整奖励函数

---

## 🚀 立即开始

```bash
# 1. 确保使用最新代码
git status  # 检查是否有未提交的更改

# 2. 删除旧模型（可选）
del ppo_snake_v1.zip

# 3. 开始训练
python snake.py train

# 4. 等待训练完成（4-8 小时）

# 5. 测试性能
python test_ai_performance.py
```

---

## 📝 训练日志示例

良好的训练日志应该显示：

```
迭代 100 (200k 步):
  ep_len_mean: 60
  ep_rew_mean: 5.2

迭代 200 (400k 步):
  ep_len_mean: 95
  ep_rew_mean: 12.8

迭代 300 (600k 步):
  ep_len_mean: 125
  ep_rew_mean: 18.5

迭代 500 (1M 步):
  ep_len_mean: 145
  ep_rew_mean: 22.3

迭代 1000 (2M 步):
  ep_len_mean: 165
  ep_rew_mean: 26.7  ← 目标达成！
```

如果看到这样的趋势，说明训练进展顺利！✅

---

**加油！耐心训练，AI 会越来越聪明！** 🤖🐍✨