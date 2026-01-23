from stable_baselines3 import PPO
import gymnasium as gym
import time
env = gym.make("Ant-v4", render_mode="human")
model = PPO.load("ant_walker")

obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        obs, _ = env.reset()