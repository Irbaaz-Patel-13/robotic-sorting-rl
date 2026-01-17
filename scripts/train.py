#!/usr/bin/env python3

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# These environment variables help with threading issues
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
import gymnasium as gym

torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

import panda_gym
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Try importing TQC, fall back to SAC if not available
try:
    from sb3_contrib import TQC
    HAS_TQC = True
except ImportError:
    print("Note: sb3-contrib not installed, using SAC instead of TQC")
    print("For better results, install it: pip install sb3-contrib")
    from stable_baselines3 import SAC as TQC
    HAS_TQC = False

from stable_baselines3 import HerReplayBuffer


class ProgressCallback(BaseCallback):
    """
    Prints training progress every N steps.
    
    I added this because watching TensorBoard all the time gets tiring,
    and it's nice to see progress in the terminal.
    """
    
    def __init__(self, check_freq=5000, total_timesteps=1000000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.best_success = 0
        
    def _on_training_start(self):
        self.start_time = time.time()
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get current metrics from logger
            success_rate = self.logger.name_to_value.get('rollout/success_rate', 0)
            mean_reward = self.logger.name_to_value.get('rollout/ep_rew_mean', 0)
            
            # Track best success rate
            if success_rate > self.best_success:
                self.best_success = success_rate
            
            # Calculate timing
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            
            if progress > 0:
                eta = elapsed / progress - elapsed
                eta_str = f"{eta/60:.1f}min"
            else:
                eta_str = "calculating..."
            
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Print progress bar
            print("\n" + "=" * 60)
            print(f"Step: {self.num_timesteps:,} / {self.total_timesteps:,} ({progress*100:.1f}%)")
            print(f"Success Rate: {success_rate*100:.1f}% (best: {self.best_success*100:.1f}%)")
            print(f"Mean Reward: {mean_reward:.2f}")
            print(f"FPS: {fps:.1f} | Elapsed: {elapsed/60:.1f}min | ETA: {eta_str}")
            print("=" * 60)
            
        return True


def create_env(env_id, reward_type='sparse'):
    """Create and wrap the environment."""
    env = gym.make(env_id, reward_type=reward_type)
    env = Monitor(env)
    return env


def make_train_env(env_id, reward_type='sparse'):
    """Create training environment wrapped in DummyVecEnv."""
    def _init():
        return create_env(env_id, reward_type)
    return DummyVecEnv([_init])


def train(args):
    """Main training function."""
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tqc_{args.reward_type}_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("ROBOTIC PICK-AND-PLACE TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Algorithm: {'TQC' if HAS_TQC else 'SAC'}")
    print(f"  Reward Type: {args.reward_type}")
    print(f"  Use HER: {args.use_her}")
    print(f"  Total Timesteps: {args.total_timesteps:,}")
    print(f"  Device: {args.device}")
    print(f"  Log Directory: {log_dir}")
    print("=" * 70 + "\n")
    
    # Create environments
    print("Creating environments...")
    train_env = make_train_env(args.env_id, args.reward_type)
    eval_env = create_env(args.env_id, args.reward_type)
    
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    
    # Set up the model
    print("\nSetting up model...")
    
    # These hyperparameters were tuned through lots of trial and error
    # The key ones are gamma=0.95 and batch_size=512
    model_kwargs = dict(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=1000,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
        ),
        tensorboard_log=str(log_dir / "tensorboard"),
        device=args.device,
        verbose=1,
    )
    
    # Add TQC-specific parameters if available
    if HAS_TQC:
        model_kwargs["policy_kwargs"]["n_critics"] = 2
        model_kwargs["policy_kwargs"]["n_quantiles"] = 25
        model_kwargs["top_quantiles_to_drop_per_net"] = 2
    
    # Add HER if using sparse rewards
    if args.use_her and args.reward_type == 'sparse':
        model_kwargs["replay_buffer_class"] = HerReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )
    
    # Load checkpoint or create new model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        model = TQC.load(args.checkpoint, env=train_env, **model_kwargs)
    else:
        print("Creating new model")
        model = TQC(**model_kwargs)
    
    print(f"Using {model.device} device")
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="model",
        save_replay_buffer=False,  # Saves disk space
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "checkpoints" / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,
    )
    
    progress_callback = ProgressCallback(
        check_freq=args.log_freq,
        total_timesteps=args.total_timesteps,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback, progress_callback])
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")
    
    # Save final model
    final_path = log_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")
    
    # Run final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    successes = 0
    rewards = []
    
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        if info.get('is_success', False):
            successes += 1
        rewards.append(ep_reward)
    
    print(f"\nResults over 20 episodes:")
    print(f"  Success Rate: {successes/20*100:.1f}%")
    print(f"  Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Min/Max Reward: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Logs saved to: {log_dir}")
    print("=" * 70 + "\n")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, log_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a robot arm for pick-and-place tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment settings
    parser.add_argument('--env-id', type=str, default='PandaPickAndPlace-v3',
                        help='Gymnasium environment ID')
    parser.add_argument('--reward-type', type=str, default='sparse',
                        choices=['sparse', 'dense'],
                        help='Reward type (sparse requires HER)')
    
    # Algorithm settings
    parser.add_argument('--use-her', action='store_true', default=True,
                        help='Use Hindsight Experience Replay')
    parser.add_argument('--no-her', action='store_false', dest='use_her',
                        help='Disable HER')
    
    # Hyperparameters (these defaults work well)
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor (0.95 works better than 0.99 for manipulation)')
    parser.add_argument('--tau', type=float, default=0.05,
                        help='Soft update coefficient')
    
    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging settings
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--log-freq', type=int, default=5000,
                        help='How often to print progress')
    parser.add_argument('--eval-freq', type=int, default=25000,
                        help='How often to run evaluation')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='How often to save checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for training (auto, cuda, cpu)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
