# fastapi-starter

## features

1. pytest & pytest-cov
2. pre-commit
3. darker for code format
4. ruff for lint
5. commitlint for commit message format
6. typer for cli
7. sqlmodel for database ORM
8. alembic for database migration
9. preset middlewares, including:
    1. [slowapi](https://slowapi.readthedocs.io/en/latest/) for rate limit
    2. [fastapi-users](https://fastapi-users.github.io/fastapi-users) for auth
    3. [cors](https://fastapi.tiangolo.com/tutorial/cors/) for cross-origin resource sharing
    4. [fastapi-pagination](https://uriyyo-fastapi-pagination.netlify.app/) for pagination

## introduction

the project uses `src-layout`, includes `my_sdk` (for publish) and `biz` (for business logic).

~~`sdk` is designed for publish and no need to rename, we use file/folder mapping `my_sdk=src/sdk` in `pyproject.toml`.~~

feel free to rename `my_sdk` to `[the name you want]` in the whole project for your own use or publish.

### dev

before you start:

1. must run `./scripts/setup`
1. replace `my_sdk` with `[the name you want]` in the whole project
1. ~~remember to `uv pip install -e .[all]` to make sure `my_sdk` is available in development.~~

### unit test

`uv run -m pytest`

> `htmlcov` folder contains html coverage report.

### run cli

`uv run main.py --help`

### run server

`uv run main.py serve`

> if you have `html` files need to `serve`, please put them in `static` folder.

### release sdk

1. `cp .pypirc.example .pypirc`
2. replace your token in `.pypirc`
3. `git tag vx.y.z`
4. `./scripts/publish`

### build executable file

`./scripts/build`

---

## 🐍 贪吃蛇游戏 (Snake Game)

本项目包含一个基于强化学习的贪吃蛇游戏，支持 **AI 自动训练和游玩**以及人类玩家模式。

### 🎯 主要功能

- ✅ **AI 自动训练**: 使用 PPO 算法让 AI 学习玩贪吃蛇
- ✅ **AI 自动游玩**: 加载训练好的模型，观看 AI 表演
- ✅ **人类玩家**: 使用方向键自己控制蛇
- ✅ **实时可视化**: Pygame 图形界面展示
- ✅ **性能统计**: 记录得分、最高分等数据

### 📦 安装依赖

```bash
pip install gymnasium pygame stable-baselines3 numpy
```

或使用 uv:
```bash
uv pip install gymnasium pygame stable-baselines3 numpy
```

### 🚀 快速开始

#### 方式一：交互菜单（推荐）
```bash
python snake.py
```
然后根据提示选择模式（1/2/3）

#### 方式二：命令行参数
```bash
# 训练 AI
python snake.py train

# 测试 AI（观看 AI 玩游戏）
python snake.py test

# 人类玩家模式
python snake.py play
```

#### 方式三：快速启动脚本
```bash
# 智能判断：有模型则测试，无模型则训练
python run_ai.py

# 或指定模式
python run_ai.py train
python run_ai.py test
```

### 🎮 游戏模式详解

#### 1. 训练 AI (train)
使用 PPO 强化学习算法训练 AI 玩贪吃蛇。

**⚠️ 重要更新（v2.1）**：
- ✅ 优化了观察空间：新增左/右危险检测和食物方向信息
- ✅ 优化了奖励函数：添加距离引导，防止 AI 转圈
- ✅ 优化了训练参数：增加探索能力和网络容量
- **需要重新训练**：旧模型不再适用，请删除 `ppo_snake_v1.zip` 后重新训练

**特点**：
- 训练步数：1,000,000 步
- 自动保存模型为 `ppo_snake_v1.zip`
- 训练日志保存在 `./logs/` 目录
- 支持 TensorBoard 可视化
- 可按 `Ctrl+C` 随时停止

**训练时间**：
- CPU: 约 2-4 小时
- GPU: 约 30-60 分钟

**监控训练**：
```bash
tensorboard --logdir=./logs/
# 浏览器访问 http://localhost:6006
```

**预期效果**：
- 修复前：AI 原地转圈，得分 2-5
- 修复后：AI 主动寻食，得分 8-15+

#### 2. AI 测试 (test)
加载已训练的 AI 模型，观看 AI 自动玩游戏。

**前提条件**：
- 需要先运行训练模式生成 `ppo_snake_v1.zip`

**功能**：
- 实时图形界面展示
- 自动进行多局游戏
- 显示每局得分
- 追踪最高得分纪录
- 帧率：15 FPS（流畅观看）

**退出方式**：
- 按 `ESC` 键
- 或关闭窗口

#### 3. 人类玩家模式 (play)
使用方向键控制蛇移动，自己玩贪吃蛇。

**控制方式**：
- ⬆️⬇️⬅️➡️ 方向键控制移动
- 任意键：游戏结束后重新开始
- `ESC`：退出游戏

**游戏规则**：
- 吃到红色食物增加长度和得分
- 撞墙或撞到自己游戏结束
- 游戏结束后按任意键重新开始

### 🧪 运行测试

```bash
python test_snake.py
```

这将运行自动化测试，验证：
- ✅ 环境初始化
- ✅ 观察空间维度
- ✅ 碰撞检测
- ✅ 游戏逻辑

### 🤖 AI 技术细节

#### 强化学习框架
- **算法**: PPO (Proximal Policy Optimization)
- **框架**: Stable-Baselines3
- **策略网络**: MLP (多层感知机)

#### 观察空间 (8维向量)
AI "看到"的信息：
1. 蛇头 X 坐标（归一化 0-1）
2. 蛇头 Y 坐标（归一化 0-1）
3. 食物 X 坐标（归一化 0-1）
4. 食物 Y 坐标（归一化 0-1）
5. 前方是否危险（撞墙/撞自己）
6. 预留维度（可扩展）
7. 预留维度（可扩展）
8. 后方是否危险

#### 动作空间 (3种)
- **0**: 直走（保持当前方向）
- **1**: 右转（顺时针 90°）
- **2**: 左转（逆时针 90°）

#### 奖励函数
| 事件 | 奖励 | 说明 |
|------|------|------|
| 吃到食物 | +10 | 鼓励寻找食物 |
| 死亡 | -10 | 避免危险行为 |
| 每步移动 | -0.1 | 鼓励快速行动 |
| 超时死亡 | -5 | 防止苟活策略 |

### 📊 预期效果

| 训练步数 | 平均得分 | 最高得分 | 训练时间 (CPU) |
|---------|---------|---------|--------------|
| 100,000 | 3-5 | 8-10 | ~20 分钟 |
| 500,000 | 8-12 | 15-20 | ~1-2 小时 |
| 1,000,000 | 12-18 | 25-35 | ~3-4 小时 |

*注：具体数值因环境和超参数而异*

### 📚 更多文档

- [AI 详细使用指南](AI_GUIDE.md) - 完整的 AI 训练和测试说明
- [AI 实现总结](AI_IMPLEMENTATION_SUMMARY.md) - 技术实现细节
- [人类玩家模式说明](SNAKE_HUMAN_MODE.md) - 人类玩家模式优化说明
- [问题修复记录](SNAKE_FIX_NOTES.md) - 已知问题和修复方案
- [快速参考](SNAKE_QUICK_REF.md) - 快速上手指南

### 💡 使用建议

1. **首次使用**：先运行训练 `python snake.py train`
2. **等待训练完成**：可能需要几十分钟到几小时
3. **测试 AI**：运行 `python snake.py test` 观看 AI 表演
4. **调整参数**：根据表现调整超参数重新训练
5. **人类娱乐**：运行 `python snake.py play` 自己玩

### 🔧 常见问题

**Q: AI 为什么总是撞墙？**  
A: 训练不足，请增加训练步数或检查奖励函数。

**Q: 如何加速训练？**  
A: 使用 GPU、减少日志输出、并行多个环境。

**Q: 找不到模型文件？**  
A: 请先运行训练模式生成 `ppo_snake_v1.zip`。

**Q: AI 表现不稳定？**  
A: 继续训练更多步数，或调整探索系数。

更多问题请查看 [AI_GUIDE.md](AI_GUIDE.md) 的常见问题部分。

---

**版本**: v2.0 (AI 完整版)  
**最后更新**: 2026-05-09
