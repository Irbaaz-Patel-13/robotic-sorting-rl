#!/usr/bin/env python3

import argparse
import numpy as np
import gymnasium as gym
import panda_gym
from pathlib import Path

try:
    from sb3_contrib import TQC
    HAS_TQC = True
except ImportError:
    HAS_TQC = False

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


def load_model(checkpoint_path, env):
    """
    Load a trained model from checkpoint.
    Tries TQC first, falls back to SAC if that fails.
    """
    checkpoint_path = str(checkpoint_path)
    
    # Remove .zip if already present (SB3 adds it automatically)
    if checkpoint_path.endswith('.zip'):
        checkpoint_path = checkpoint_path[:-4]
    
    if HAS_TQC:
        try:
            return TQC.load(checkpoint_path, env=env)
        except Exception as e:
            print(f"Couldn't load as TQC: {e}")
    
    return SAC.load(checkpoint_path, env=env)


def evaluate(args):
    """Run evaluation on a trained model."""
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env_id}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60 + "\n")
    
    # Create environment
    env = gym.make(args.env_id, reward_type='sparse')
    
    # Wrap in DummyVecEnv for compatibility with SB3
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, vec_env)
    print(f"Model loaded successfully\n")
    
    # Run evaluation
    successes = 0
    rewards = []
    episode_lengths = []
    
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        
        success = info.get('is_success', False)
        if success:
            successes += 1
        
        rewards.append(ep_reward)
        episode_lengths.append(steps)
        
        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0:
            print(f"Progress: {ep + 1}/{args.n_episodes} episodes "
                  f"(current success rate: {successes/(ep+1)*100:.1f}%)")
    
    env.close()
    
    # Calculate statistics
    success_rate = successes / args.n_episodes * 100
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate:    {success_rate:.1f}% ({successes}/{args.n_episodes})")
    print(f"Mean Reward:     {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode Length:  {mean_length:.1f} +/- {std_length:.1f}")
    print(f"Min/Max Reward:  {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print("=" * 60 + "\n")
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Environment: {args.env_id}\n")
            f.write(f"Episodes: {args.n_episodes}\n\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
            f.write(f"Episode Length: {mean_length:.1f} +/- {std_length:.1f}\n")
            f.write(f"\nPer-episode rewards:\n")
            for i, r in enumerate(rewards):
                f.write(f"  Episode {i+1}: {r:.2f}\n")
        
        print(f"Results saved to: {output_path}")
    
    return {
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'rewards': rewards,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained pick-and-place model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--env-id', type=str, default='PandaPickAndPlace-v3',
                        help='Environment ID')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results (optional)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
