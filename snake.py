import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os


# --- 1. 定义自定义环境 (Custom Environment) ---
class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        # 游戏参数
        self.grid_size = 20  # 网格大小 (20x20)
        self.block_size = 20  # 像素大小
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size

        self.render_mode = render_mode

        # 动作空间: 0=直走, 1=右转, 2=左转 (Discrete)
        self.action_space = spaces.Discrete(3)

        # 观察空间: 我们使用一个简单的向量表示状态
        # [蛇头X, 蛇头Y, 食物X, 食物Y, 危险直走, 危险左转, 危险右转]
        # 这里为了简化演示，我们只传回核心坐标信息，实际复杂项目可用 CNN 处理图像
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(8,), dtype=np.float32  # 8个特征
        )

        # 初始化 Pygame (仅用于渲染)
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("SB3 Snake AI")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置游戏状态
        self.snake = [(10, 10), (9, 10), (8, 10)]  # 蛇身坐标 (网格坐标)
        self.direction = (1, 0)  # 初始向右
        self.food = self._place_food()
        self.steps_without_food = 0
        self.max_steps = 100 * self.grid_size  # 防止死循环

        return self._get_obs(), {}

    def _place_food(self):
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food not in self.snake:
                return food

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # 计算蛇头前方、左方、右方的坐标
        next_x = head_x + self.direction[0]
        next_y = head_y + self.direction[1]
        
        # 左转后的方向
        left_dir = (-self.direction[1], self.direction[0])
        left_x = head_x + left_dir[0]
        left_y = head_y + left_dir[1]
        
        # 右转后的方向
        right_dir = (self.direction[1], -self.direction[0])
        right_x = head_x + right_dir[0]
        right_y = head_y + right_dir[1]

        # 计算食物相对于蛇头的方向
        dx = food_x - head_x
        dy = food_y - head_y
        
        # 判断食物在哪个方向（相对于当前朝向）
        # 使用点积判断前后，叉积判断左右
        forward_dot = dx * self.direction[0] + dy * self.direction[1]
        
        # 归一化到观察空间
        obs = np.array(
            [
                head_x / self.grid_size,  # 1. 蛇头X (归一化)
                head_y / self.grid_size,  # 2. 蛇头Y (归一化)
                food_x / self.grid_size,  # 3. 食物X (归一化)
                food_y / self.grid_size,  # 4. 食物Y (归一化)
                self._is_collision((next_x, next_y)),  # 5. 前方危险
                self._is_collision((left_x, left_y)),  # 6. 左方危险
                self._is_collision((right_x, right_y)),  # 7. 右方危险
                forward_dot / (self.grid_size * 1.414),  # 8. 食物在前方的投影（正=前，负=后）
            ],
            dtype=np.float32,
        )

        return obs

    def _is_collision(self, pos):
        x, y = pos
        # 撞墙 或 撞自己
        if (
            x < 0
            or x >= self.grid_size
            or y < 0
            or y >= self.grid_size
            or pos in self.snake
        ):
            return 1.0
        return 0.0

    def step(self, action):
        # 记录移动前的位置，用于计算奖励
        old_head = self.snake[0]
        old_distance_to_food = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        
        # 动作处理: 0=直走, 1=右转, 2=左转
        if action == 1:  # 右转
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # 左转
            self.direction = (-self.direction[1], self.direction[0])

        # 移动蛇头
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # --- 奖励设计 (Reward Shaping) ---
        reward = 0
        terminated = False
        info = {}

        # 1. 撞墙或撞自己 -> 死亡（大惩罚）
        if self._is_collision(new_head):
            reward = -10
            terminated = True
            return self._get_obs(), reward, terminated, False, info

        self.snake.insert(0, new_head)

        # 2. 吃到食物（大奖励）
        if new_head == self.food:
            reward = 10
            self.food = self._place_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()  # 没吃到，移除尾巴
            
            # 3. 距离奖励：鼓励向食物移动
            new_distance_to_food = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            distance_change = old_distance_to_food - new_distance_to_food
            
            if distance_change > 0:
                # 向食物靠近，给予小奖励
                reward = 0.1
            elif distance_change < 0:
                # 远离食物，给予小惩罚
                reward = -0.2
            else:
                # 距离不变，轻微惩罚防止原地转圈
                reward = -0.05
            
            self.steps_without_food += 1

        # 4. 步数限制：防止 AI 学会"苟活"而不吃食物
        if self.steps_without_food > self.max_steps:
            terminated = True
            reward = -5

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))  # 黑色背景
            # 画蛇
            for pt in self.snake:
                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    (
                        pt[0] * self.block_size,
                        pt[1] * self.block_size,
                        self.block_size - 2,
                        self.block_size - 2,
                    ),
                )
            # 画食物
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (
                    self.food[0] * self.block_size,
                    self.food[1] * self.block_size,
                    self.block_size - 2,
                    self.block_size - 2,
                ),
            )
            pygame.display.flip()


# --- 2. 训练主程序 ---
def train():
    """训练 AI 模型"""
    print("=== 开始训练 AI ===")
    print("这将使用 PPO 算法训练贪吃蛇 AI")
    print("训练过程可能需要几分钟到几小时，取决于你的电脑性能\n")
    
    # 创建环境
    env = SnakeEnv()

    # 检查环境是否符合 Gym 标准 (可选，但在开发自定义环境时很有用)
    # check_env(env, warn=True)

    # 初始化 PPO 模型
    # MlpPolicy 表示使用多层感知机，适合处理向量输入
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
        ent_coef=0.03,  # 从 0.02 增加到 0.03，鼓励更多探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
    )

    print("开始训练... 按 Ctrl+C 停止")
    print("💡 优化配置:")
    print("  • 探索系数: 0.03 (增加探索)")
    print("  • 训练步数: 2,000,000 (充分训练)")
    print("  • 预期效果: 平均得分 12-18，最高得分 20-30\n")
    try:
        # 训练 2,000,000 步 (从 100万 增加到 200万)
        model.learn(total_timesteps=2000000)
    except KeyboardInterrupt:
        print("\n训练被用户中断")

    # 保存模型
    model.save("ppo_snake_v1")
    print("\n✅ 模型已保存为 ppo_snake_v1.zip")
    print("💡 提示：可以使用 'python snake.py test' 来测试训练好的模型")


def test():
    """测试 AI 模型 - AI 自动玩游戏"""
    print("=== AI 自动游玩模式 ===")
    print("加载训练好的模型...\n")
    
    if not os.path.exists("ppo_snake_v1.zip"):
        print("❌ 未找到模型文件 ppo_snake_v1.zip")
        print("请先运行训练模式: python snake.py train")
        return

    # 加载模型
    env = SnakeEnv(render_mode="human")
    model = PPO.load("ppo_snake_v1", env=env)
    
    print("✅ 模型加载成功！")
    print("AI 将自动控制蛇移动")
    print("按 ESC 或关闭窗口退出\n")

    clock = pygame.time.Clock()
    running = True
    episode = 0
    best_score = 0

    while running:
        obs, _ = env.reset()
        episode += 1
        terminated = False
        truncated = False
        
        print(f"--- 第 {episode} 局开始 ---")

        while not terminated and not truncated and running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if not running:
                break

            # AI 预测动作 (deterministic=True 表示使用确定性策略)
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染画面
            env.render()
            
            # 控制帧率
            clock.tick(15)  # 15 FPS，让 AI 玩得更快

        # 游戏结束
        if running:
            score = len(env.snake) - 3
            if score > best_score:
                best_score = score
                print(f"🎉 新纪录！得分: {score}")
            else:
                print(f"游戏结束！得分: {score} (最高: {best_score})")
            
            # 短暂暂停后开始下一局
            pygame.time.wait(1000)

    pygame.quit()
    print(f"\nAI 测试结束！共玩了 {episode} 局，最高得分: {best_score}")


def play_human():
    print("人类玩家模式！使用方向键控制蛇的移动")
    print("控制方式: ↑↓←→ 方向键")
    print("按 ESC 或关闭窗口退出\n")
    
    env = SnakeEnv(render_mode="human")
    obs, _ = env.reset()

    clock = pygame.time.Clock()
    running = True
    game_over = False
    wait_for_start = False

    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif wait_for_start:
                    # 游戏结束后，按任意键重新开始
                    wait_for_start = False
                    game_over = False
                    obs, _ = env.reset()
                    print("新游戏开始！\n")
                else:
                    # 正常游戏中，处理方向控制
                    current_dir = env.direction
                    
                    # 根据按键更新方向（防止180度转向）
                    if event.key == pygame.K_UP and current_dir != (0, 1):
                        env.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and current_dir != (0, -1):
                        env.direction = (0, 1)
                    elif event.key == pygame.K_LEFT and current_dir != (1, 0):
                        env.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and current_dir != (-1, 0):
                        env.direction = (1, 0)

        if wait_for_start:
            # 游戏结束状态，显示提示并等待按键
            font = pygame.font.Font(None, 48)
            text = font.render("Press Any Key to Restart", True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.width // 2, env.height // 2))
            env.screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(10)
            continue

        # 执行一步游戏（动作0表示按当前方向直走）
        obs, reward, terminated, truncated, info = env.step(0)
        
        # 渲染画面
        env.render()
        
        # 检查游戏是否结束
        if terminated or truncated:
            score = len(env.snake) - 3
            print(f"游戏结束！得分: {score}")
            
            # 显示游戏结束提示
            font = pygame.font.Font(None, 48)
            text1 = font.render(f"Game Over! Score: {score}", True, (255, 255, 255))
            text2 = font.render("Press Any Key", True, (255, 255, 255))
            
            text1_rect = text1.get_rect(center=(env.width // 2, env.height // 2 - 30))
            text2_rect = text2.get_rect(center=(env.width // 2, env.height // 2 + 30))
            
            env.screen.blit(text1, text1_rect)
            env.screen.blit(text2, text2_rect)
            pygame.display.flip()
            
            # 进入等待状态，不按自动重新开始
            wait_for_start = True
            game_over = True

        # 控制游戏速度为 8 FPS（更适合人类玩家）
        clock.tick(8)

    pygame.quit()
    print("游戏已退出")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train" or mode == "1":
            train()
        elif mode == "test" or mode == "2":
            test()
        elif mode == "play" or mode == "3":
            play_human()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: train, test, play")
            print("或直接运行 python snake.py 进入交互模式")
    else:
        # 交互式选择模式
        print("=" * 50)
        print("       🐍 贪吃蛇游戏 - Snake Game 🐍")
        print("=" * 50)
        print()
        print("请选择模式:")
        print("  1. 训练 AI    - 使用 PPO 算法训练 AI 玩贪吃蛇")
        print("  2. AI 测试    - 观看 AI 自动玩游戏（需要已训练）")
        print("  3. 人类玩家   - 使用方向键自己控制蛇")
        print()
        print("=" * 50)

        choice = input("请输入选择 (1/2/3): ").strip()

        if choice == "1" or choice.lower() == "train":
            train()
        elif choice == "2" or choice.lower() == "test":
            test()
        elif choice == "3" or choice.lower() == "play":
            play_human()
        else:
            print("无效选择，默认进入人类玩家模式")
            play_human()
