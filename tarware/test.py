import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomCallback(BaseCallback):
    """
    Custom callback for monitoring and logging training progress.
    """
    def __init__(self, verbose=0, log_freq=10):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.start_time = time.time()
        self.total_steps = 0
        
        # Create directories for saving results if they don't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def _on_step(self):
        # Log episode info if available
        if self.locals.get("dones") and np.any(self.locals.get("dones")):
            self.episode_rewards.append(self.locals.get("rewards"))
            self.episode_lengths.append(self.locals.get("n_steps"))
            self.episode_times.append(time.time() - self.start_time)
            self.total_steps += self.locals.get("n_steps")
            
            # Print episode info every log_freq episodes
            if len(self.episode_rewards) % self.log_freq == 0:
                avg_reward = np.mean([np.sum(r) for r in self.episode_rewards[-self.log_freq:]])
                avg_length = np.mean(self.episode_lengths[-self.log_freq:])
                elapsed_time = time.time() - self.start_time
                fps = self.total_steps / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Episode: {len(self.episode_rewards)} | " +
                      f"Avg reward: {avg_reward:.2f} | " +
                      f"Avg length: {avg_length:.2f} | " +
                      f"FPS: {fps:.2f} | " +
                      f"Elapsed time: {elapsed_time:.2f}s")
                
                # Save model checkpoint
                if len(self.episode_rewards) % (self.log_freq * 10) == 0:
                    self.model.save(f"models/ppo_tarware_{len(self.episode_rewards)}")
                    self._plot_training()
        
        return True
    
    def _plot_training(self):
        """Plot training progress."""
        plt.figure(figsize=(12, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        rewards = [np.sum(r) for r in self.episode_rewards]
        plt.plot(rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        # Plot moving average of rewards
        plt.subplot(2, 2, 2)
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(moving_avg)
        plt.title("Moving Average of Rewards (window=10)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        
        # Plot episode lengths
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        
        # Plot training speed (FPS)
        plt.subplot(2, 2, 4)
        fps = []
        for i in range(1, len(self.episode_times)):
            steps = sum(self.episode_lengths[:i])
            fps.append(steps / self.episode_times[i])
        plt.plot(fps)
        plt.title("Training Speed (FPS)")
        plt.xlabel("Episode")
        plt.ylabel("Frames Per Second")
        
        plt.tight_layout()
        plt.savefig(f"logs/training_progress_{len(self.episode_rewards)}.png")
        plt.close()

def print_env_info(env):
    """Print detailed information about the environment."""
    print("\n=========== ENVIRONMENT INFORMATION ===========")
    print(f"Environment ID: {env.unwrapped.spec.id}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Print environment-specific parameters if available
    try:
        print(f"Grid Size: {env.unwrapped.grid_size}")
        print(f"Number of AGVs: {env.unwrapped.num_agvs}")
        print(f"Number of Pickers: {env.unwrapped.num_pickers}")
        print(f"Number of Shelves: {len(env.unwrapped.shelfs)}")
        print(f"Reward Type: {env.unwrapped.reward_type}")
        print(f"Max Steps: {env.unwrapped.max_steps}")
    except AttributeError:
        print("Could not access some environment attributes")
    
    print("================================================\n")

def explore_observation(obs):
    """Analyze and print information about the observation."""
    print("\n=========== OBSERVATION ANALYSIS ===========")
    
    if isinstance(obs, tuple):
        print(f"Observation is a tuple with {len(obs)} elements")
        for i, agent_obs in enumerate(obs):
            print(f"Agent {i} observation shape: {agent_obs.shape}")
            print(f"Agent {i} observation range: [{np.min(agent_obs)}, {np.max(agent_obs)}]")
            print(f"Agent {i} observation sample: {agent_obs[:5]}")
    elif isinstance(obs, np.ndarray):
        print(f"Observation shape: {obs.shape}")
        print(f"Observation range: [{np.min(obs)}, {np.max(obs)}]")
        print(f"Observation sample: {obs[:5]}")
    else:
        print(f"Observation type: {type(obs)}")
    
    print("===========================================\n")

def visualize_episode(env, model, max_steps=200):
    """Run an episode with the trained model and visualize it."""
    print("\n=========== VISUALIZATION ===========")
    obs = env.reset()[0]
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step} | Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Step {step} | Reward: {reward}")
        
        if info:
            print(f"Step {step} | Info: {info}")
        
        if terminated or truncated:
            print(f"Episode ended after {step+1} steps")
            break
    
    print("====================================\n")
    env.close()

def main():
    # Create and monitor the environment
    env_id = "tarware-tiny-2agvs-2pickers-globalobs-v1"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)
    env = Monitor(env, "logs/")
    
    # Print environment information
    print_env_info(env)
    
    # Take a random action and analyze the observation
    obs = env.reset()[0]
    print("Initial observation:")
    explore_observation(obs)
    
    action = env.action_space.sample()
    print(f"Random action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Resulting reward: {reward}")
    print(f"Info: {info}")
    explore_observation(obs)
    
    # Define model hyperparameters
    model_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "verbose": 1
    }
    
    print("\n=========== MODEL PARAMETERS ===========")
    for key, value in model_params.items():
        print(f"{key}: {value}")
    print("========================================\n")
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, **model_params)
    
    # Create custom callback
    callback = CustomCallback(verbose=1, log_freq=5)
    
    # Train the model
    print("\n=========== STARTING TRAINING ===========")
    print(f"Training for 1000 epochs...")
    start_time = time.time()
    
    # Train for 1000 epochs (we use total_timesteps as the equivalent)
    # Each "epoch" will be roughly env.max_steps steps
    total_timesteps = env.unwrapped.max_steps * 1000
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print("==========================================\n")
    
    # Save the final model
    model.save("models/ppo_tarware_final")
    print("Final model saved as models/ppo_tarware_final")
    
    # Visualize a few episodes with the trained model
    print("\nVisualizing trained agent behavior...")
    vis_env = gym.make(env_id, render_mode="human")
    for episode in range(3):
        print(f"\nVisualization Episode {episode+1}/3")
        visualize_episode(vis_env, model)
    
    # Close environments
    env.close()
    vis_env.close()

if __name__ == "__main__":
    main()