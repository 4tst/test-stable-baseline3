# 贪吃蛇 AI - 完整使用指南

## 🤖 AI 自动游玩实现

本项目的贪吃蛇游戏已经实现了基于 **PPO (Proximal Policy Optimization)** 强化学习算法的 AI，可以自动玩贪吃蛇游戏。

## 📋 目录

1. [快速开始](#快速开始)
2. [训练 AI](#训练-ai)
3. [测试 AI](#测试-ai)
4. [AI 工作原理](#ai-工作原理)
5. [调优建议](#调优建议)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方式一：命令行启动

```bash
# 训练 AI
python snake.py train

# 测试 AI（观看 AI 玩游戏）
python snake.py test

# 人类玩家模式
python snake.py play
```

### 方式二：交互菜单

```bash
python snake.py
```

然后根据提示选择模式（输入 1/2/3）。

---

## 🎓 训练 AI

### 启动训练

```bash
python snake.py train
```

### 训练过程

1. **初始化环境**：创建 20x20 的贪吃蛇游戏环境
2. **创建 PPO 模型**：使用多层感知机（MLP）策略
3. **开始训练**：AI 通过试错学习如何玩贪吃蛇
4. **保存模型**：训练完成后保存为 `ppo_snake_v1.zip`

### 训练参数

当前配置的超参数：
- **总步数**: 500,000 步
- **学习率**: 3e-4
- **批量大小**: 64
- **训练轮数**: 10 epochs
- **折扣因子**: 0.99
- **探索系数**: 0.01

### 训练时间

- **CPU**: 约 30 分钟 - 2 小时
- **GPU**: 约 10 - 30 分钟
- 取决于你的电脑性能和训练步数

### 监控训练

训练过程中会显示：
- 当前步数
- 累计奖励
- 策略损失
- 价值函数损失

你可以按 `Ctrl+C` 随时停止训练，已训练的进度会被保存。

### TensorBoard 可视化

训练日志保存在 `./logs/` 目录，可以使用 TensorBoard 查看：

```bash
tensorboard --logdir=./logs/
```

然后在浏览器中打开 `http://localhost:6006`

---

## 🧪 测试 AI

### 前提条件

必须先完成训练，生成 `ppo_snake_v1.zip` 文件。

### 启动测试

```bash
python snake.py test
```

### AI 测试功能

1. **自动加载模型**：从 `ppo_snake_v1.zip` 加载训练好的 AI
2. **实时渲染**：以图形界面展示 AI 的游戏过程
3. **多局统计**：
   - 记录每局得分
   - 追踪最高得分
   - 显示总局数
4. **帧率控制**：15 FPS，让 AI 玩得更快更流畅

### 测试界面

- **绿色方块**：蛇身
- **红色方块**：食物
- **黑色背景**：游戏区域
- **控制台输出**：显示每局得分和最高纪录

### 退出测试

- 按 **ESC** 键
- 或关闭窗口

---

## 🧠 AI 工作原理

### 强化学习框架

```
┌─────────────┐
│  环境状态    │ ← 观察空间 (8维向量)
│  (State)    │   - 蛇头位置 (X, Y)
└──────┬──────┘   - 食物位置 (X, Y)
       │          - 危险检测 (前方、后方)
       ▼
┌─────────────┐
│   AI 大脑    │ ← PPO 神经网络
│  (Policy)   │   输入: 状态向量
└──────┬──────┘   输出: 动作概率
       │
       ▼
┌─────────────┐
│  执行动作    │ ← 动作空间 (3种)
│  (Action)   │   0: 直走
└──────┬──────┘   1: 右转
       │          2: 左转
       ▼
┌─────────────┐
│  获得奖励    │ ← 奖励函数
│  (Reward)   │   +10: 吃到食物
└──────┬──────┘   -10: 死亡
       │          -0.1: 每步时间惩罚
       └──→ 回到状态
```

### 观察空间 (8维)

AI "看到"的信息：
1. 蛇头 X 坐标（归一化 0-1）
2. 蛇头 Y 坐标（归一化 0-1）
3. 食物 X 坐标（归一化 0-1）
4. 食物 Y 坐标（归一化 0-1）
5. 前方是否危险（撞墙/撞自己）
6. 预留维度（可扩展）
7. 预留维度（可扩展）
8. 后方是否危险

### 动作空间 (3种)

AI 可以执行的动作：
- **0 - 直走**：保持当前方向
- **1 - 右转**：顺时针旋转 90 度
- **2 - 左转**：逆时针旋转 90 度

### 奖励设计

| 事件 | 奖励 | 说明 |
|------|------|------|
| 吃到食物 | +10 | 鼓励 AI 寻找食物 |
| 死亡 | -10 | 惩罚 AI 避免危险 |
| 每步移动 | -0.1 | 鼓励快速找到食物 |
| 超时死亡 | -5 | 防止 AI 学会"苟活" |

---

## ⚙️ 调优建议

### 提高 AI 性能

#### 1. 增加训练步数

```python
model.learn(total_timesteps=1000000)  # 增加到 100 万步
```

#### 2. 调整超参数

```python
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=1e-4,      # 降低学习率，更稳定
    n_steps=4096,            # 增加采样步数
    batch_size=128,          # 增大批量
    n_epochs=20,             # 增加训练轮数
    ent_coef=0.02,           # 增加探索
    ...
)
```

#### 3. 改进观察空间

可以添加更多特征：
- 到食物的距离和方向
- 周围格子的危险程度
- 蛇身长度信息
- 可用空间大小

#### 4. 使用 CNN 处理图像

如果将游戏画面作为输入：
```python
model = PPO("CnnPolicy", env, ...)
```

### 训练技巧

1. **多次训练取最优**：训练多个模型，选择表现最好的
2. **课程学习**：先在小网格训练，再迁移到大网格
3. **自对弈**：让 AI 与不同版本的自己对战
4. **数据增强**：旋转、翻转游戏状态

---

## ❓ 常见问题

### Q1: AI 为什么总是撞墙？

**原因**：训练不足或奖励设计不合理

**解决方案**：
- 增加训练步数（至少 50 万步）
- 检查奖励函数是否正确
- 确保环境没有 bug

### Q2: AI 学得有多好？

**评估指标**：
- 平均得分：> 10 分算不错
- 最高得分：> 20 分算优秀
- 存活步数：越长越好

### Q3: 如何加速训练？

**方法**：
1. 使用 GPU（CUDA）
2. 减少 verbose 输出
3. 并行训练多个环境
4. 使用向量化的环境

### Q4: 找不到模型文件？

**检查**：
```bash
ls ppo_snake_v1.zip
```

如果没有，请先运行训练：
```bash
python snake.py train
```

### Q5: AI 表现不稳定？

**可能原因**：
- 训练不充分
- 过拟合
- 探索不足

**解决方案**：
- 继续训练更多步数
- 调整探索系数 `ent_coef`
- 使用不同的随机种子

---

## 📊 性能基准

### 训练效果参考

| 训练步数 | 平均得分 | 最高得分 | 训练时间 (CPU) |
|---------|---------|---------|--------------|
| 100,000 | 3-5 | 8-10 | ~20 分钟 |
| 500,000 | 8-12 | 15-20 | ~1-2 小时 |
| 1,000,000 | 12-18 | 25-35 | ~3-4 小时 |
| 5,000,000+ | 20+ | 50+ | ~10+ 小时 |

*注：具体数值因环境和超参数而异*

---

## 🔧 高级用法

### 自定义训练脚本

```python
from snake import SnakeEnv
from stable_baselines3 import PPO

# 创建环境
env = SnakeEnv()

# 创建模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=100000)

# 保存
model.save("my_custom_model")

# 加载和使用
model = PPO.load("my_custom_model")
obs, _ = env.reset()
action, _ = model.predict(obs)
```

### 批量测试

```python
def batch_test(model_path, num_episodes=100):
    env = SnakeEnv()
    model = PPO.load(model_path, env=env)
    
    scores = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        scores.append(len(env.snake) - 3)
    
    print(f"平均得分: {np.mean(scores):.2f}")
    print(f"最高得分: {max(scores)}")
    print(f"最低得分: {min(scores)}")
```

---

## 📚 相关资源

- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [强化学习入门](https://huggingface.co/deep-rl-course/unit0/introduction)

---

## 🎯 总结

✅ **已实现功能**：
- PPO 强化学习算法
- 自动训练和测试
- 实时可视化
- 多局统计
- TensorBoard 支持

🚀 **下一步**：
1. 运行 `python snake.py train` 训练 AI
2. 等待训练完成
3. 运行 `python snake.py test` 观看 AI 表演
4. 根据需要调整超参数重新训练

祝你的 AI 玩得开心！🤖🐍