# AI 转圈问题修复说明

## 🐛 问题描述

**用户反馈**：AI 总是在原地转圈，一直不想向着目标（食物）前进。

## 🔍 问题分析

### 根本原因

1. **观察空间信息不足**
   - 第 6、7 维度固定为 0，没有提供左/右方向的危险信息
   - 缺少食物相对于蛇头方向的信息（前/后/左/右）
   - AI "看"不到应该往哪个方向走

2. **奖励函数缺乏引导**
   - 只有吃到食物才给 +10，死亡给 -10
   - 每步 -0.1 的惩罚太简单，无法引导 AI 向食物移动
   - AI 学会"苟活"策略：原地转圈避免死亡

3. **训练参数不够优化**
   - 探索系数较低 (0.01)，AI 不愿意尝试新动作
   - 网络架构较小，学习能力有限
   - 训练步数可能不足

## ✅ 修复方案

### 1. 优化观察空间 (8维)

#### 修复前
```python
obs = [
    head_x / grid_size,      # 蛇头X
    head_y / grid_size,      # 蛇头Y
    food_x / grid_size,      # 食物X
    food_y / grid_size,      # 食物Y
    is_collision(front),     # 前方危险
    0,                       # ❌ 固定为0，无信息
    0,                       # ❌ 固定为0，无信息
    is_collision(back),      # 后方危险
]
```

#### 修复后
```python
obs = [
    head_x / grid_size,                    # 1. 蛇头X (归一化)
    head_y / grid_size,                    # 2. 蛇头Y (归一化)
    food_x / grid_size,                    # 3. 食物X (归一化)
    food_y / grid_size,                    # 4. 食物Y (归一化)
    is_collision(front),                   # 5. 前方危险 ✅
    is_collision(left),                    # 6. 左方危险 ✅ 新增
    is_collision(right),                   # 7. 右方危险 ✅ 新增
    forward_dot / max_distance,            # 8. 食物在前方的投影 ✅ 新增
]
```

#### 改进点
- ✅ **左/右危险检测**：AI 知道左右是否有危险
- ✅ **食物方向信息**：通过点积计算食物在前方还是后方
  - 正值：食物在前方
  - 负值：食物在后方
  - AI 可以据此决定是否需要转向

### 2. 优化奖励函数

#### 修复前
```python
if 吃到食物:
    reward = +10
elif 死亡:
    reward = -10
else:
    reward = -0.1  # 简单的时间惩罚
```

**问题**：AI 学会原地转圈来避免死亡，因为 -0.1 的惩罚比 -10 小得多。

#### 修复后
```python
# 计算到食物的曼哈顿距离变化
old_distance = |head_x - food_x| + |head_y - food_y|
new_distance = |new_head_x - food_x| + |new_head_y - food_y|
distance_change = old_distance - new_distance

if 吃到食物:
    reward = +10
elif 死亡:
    reward = -10
elif distance_change > 0:  # 向食物靠近
    reward = +0.1          # ✅ 小奖励
elif distance_change < 0:  # 远离食物
    reward = -0.2          # ✅ 小惩罚
else:                      # 距离不变（原地转圈）
    reward = -0.05         # ✅ 轻微惩罚
```

**优势**：
- ✅ **正向引导**：向食物移动给予奖励
- ✅ **负向惩罚**：远离食物给予惩罚
- ✅ **防止转圈**：原地不动也有轻微惩罚

### 3. 优化训练参数

#### 主要调整

| 参数 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| ent_coef | 0.01 | **0.02** | 增加探索，鼓励尝试 |
| total_timesteps | 500,000 | **1,000,000** | 增加训练步数 |
| net_arch | 默认 | **[128, 128]** | 增大网络容量 |

```python
model = PPO(
    "MlpPolicy", 
    env,
    ent_coef=0.02,  # 从 0.01 增加到 0.02
    policy_kwargs=dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # 更大的网络
    )
)
model.learn(total_timesteps=1000000)  # 从 50万 增加到 100万
```

## 📊 预期效果对比

### 修复前
```
AI 行为：
- 原地转圈 ⭕
- 随机移动 🎲
- 不主动寻找食物 ❌
- 平均得分：2-5

原因：
- 看不到食物方向
- 没有向食物移动的激励
- 探索不足
```

### 修复后
```
AI 行为：
- 主动向食物移动 ➡️🍎
- 避开障碍物 🚧
- 高效路径规划 🗺️
- 平均得分：8-15+

原因：
- 知道食物在哪个方向
- 向食物移动有奖励
- 更强的探索能力
```

## 🚀 使用方法

### 重新训练（推荐）

由于修改了观察空间和奖励函数，**旧的模型不再适用**，需要重新训练：

```bash
# 删除旧模型（可选）
rm ppo_snake_v1.zip

# 开始新的训练
python snake.py train
```

### 训练时间预估

| 硬件 | 预计时间 |
|------|---------|
| CPU (普通) | 2-4 小时 |
| CPU (高性能) | 1-2 小时 |
| GPU | 30-60 分钟 |

### 监控训练进度

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs/

# 浏览器访问
http://localhost:6006
```

**关键指标**：
- `ep_rew_mean`：平均回合奖励（应该逐渐上升）
- `entropy`：策略熵（初期高，后期降低）

## 💡 训练技巧

### 1. 观察学习曲线

正常的学习曲线应该是：
```
奖励
  ↑
  |        /----
  |      /
  |    /
  |  /
  |/
  +----------------→ 训练步数
```

如果曲线平坦或下降，说明：
- 学习率太高/太低
- 探索不足/过度
- 奖励设计有问题

### 2. 多次训练取最优

```bash
# 第一次训练
python snake.py train
mv ppo_snake_v1.zip ppo_snake_v1_run1.zip

# 第二次训练
python snake.py train
mv ppo_snake_v1.zip ppo_snake_v1_run2.zip

# 测试比较
python snake.py test  # 手动切换模型文件测试
```

### 3. 逐步增加难度

可以先在小网格训练，再迁移到大网格：
```python
# 修改 SnakeEnv 的 grid_size
self.grid_size = 10  # 先在小网格训练
# 训练完成后
self.grid_size = 20  # 再在大网格微调
```

## 🔧 进一步优化建议

### 如果 AI 还是表现不佳

1. **继续增加训练步数**
   ```python
   model.learn(total_timesteps=2000000)  # 200万步
   ```

2. **调整学习率**
   ```python
   learning_rate=1e-4  # 降低学习率，更稳定
   ```

3. **增加批量大小**
   ```python
   batch_size=128  # 从 64 增加到 128
   ```

4. **使用向量环境**
   ```python
   from stable_baselines3.common.env_util import make_vec_env
   env = make_vec_env(lambda: SnakeEnv(), n_envs=4)
   ```

5. **添加更多观察特征**
   - 到食物的精确距离
   - 周围格子的占用情况
   - 蛇身长度信息

### 高级优化

1. **课程学习**
   - 阶段1：只有食物，无障碍
   - 阶段2：加入墙壁
   - 阶段3：加入自身碰撞

2. **自对弈**
   - 让不同版本的 AI 互相对战
   - 促进策略进化

3. **模仿学习**
   - 先用人类玩家数据预训练
   - 再用强化学习微调

## 📝 技术细节

### 观察空间数学原理

#### 食物方向计算（点积）

```python
# 蛇头到食物的向量
dx = food_x - head_x
dy = food_y - head_y

# 当前方向向量
dir_x, dir_y = self.direction

# 点积：判断食物在前方还是后方
forward_dot = dx * dir_x + dy * dir_y

# 解释：
# forward_dot > 0: 食物在前方（夹角 < 90°）
# forward_dot < 0: 食物在后方（夹角 > 90°）
# forward_dot = 0: 食物在侧面（夹角 = 90°）
```

#### 距离计算（曼哈顿距离）

```python
# 曼哈顿距离（更适合网格环境）
distance = |x1 - x2| + |y1 - y2|

# 欧几里得距离（也可用）
distance = sqrt((x1-x2)² + (y1-y2)²)
```

### 奖励函数设计原则

1. **稀疏奖励 vs 密集奖励**
   - 稀疏：只在关键事件给奖励（如吃到食物）
   - 密集：每步都给奖励引导（如距离变化）
   - **本方案采用密集奖励**，更容易学习

2. **奖励缩放**
   - 大奖励用于关键事件（±10）
   - 小奖励用于日常引导（±0.1-0.2）
   - 比例要合理，避免主导

3. **避免奖励黑客**
   - 防止 AI 找到"刷分"漏洞
   - 多重约束（时间限制、死亡惩罚等）

## ✅ 验证清单

训练完成后，检查 AI 是否学会了正确行为：

- [ ] AI 主动向食物移动
- [ ] AI 能避开墙壁
- [ ] AI 能避开自己的身体
- [ ] AI 不会长时间原地转圈
- [ ] 平均得分 > 8
- [ ] 最高得分 > 15

如果以上都满足，说明修复成功！🎉

---

**修复日期**：2026-05-09  
**版本**：v2.1 (AI 优化版)  
**状态**：✅ 已完成优化，需重新训练