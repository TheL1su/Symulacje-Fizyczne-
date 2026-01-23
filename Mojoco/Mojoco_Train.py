from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("Ant-v4")

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=1_000_000)
model.save("ant_walker")