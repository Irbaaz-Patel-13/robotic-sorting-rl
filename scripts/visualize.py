#!/usr/bin/env python3
"""
Visualization Script
====================

Watch the trained agent in action or record demo videos.

Modes:
- watch: Opens a window to watch the agent (requires display)
- record: Saves episodes to a video file

Usage:
    python scripts/visualize.py --checkpoint ./logs/best_model.zip --mode watch
    python scripts/visualize.py --checkpoint ./logs/best_model.zip --mode record --output demo.mp4

Author: Irbaaz Patel
MSc Robotics & Embedded Systems, Heriot-Watt University
"""

import argparse
import time
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

# Try to import imageio for video recording
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def load_model(checkpoint_path, env):
    """Load trained model from checkpoint."""
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith('.zip'):
        checkpoint_path = checkpoint_path[:-4]
    
    if HAS_TQC:
        try:
            return TQC.load(checkpoint_path, env=env)
        except:
            pass
    return SAC.load(checkpoint_path, env=env)


def watch_agent(args):
    """Watch the agent in real-time with rendering."""
    
    print("\n" + "=" * 60)
    print("WATCHING AGENT")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Create environment with human rendering
    env = gym.make(args.env_id, render_mode='human')
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = load_model(args.checkpoint, vec_env)
    print(f"Model loaded from {args.checkpoint}\n")
    
    episode = 0
    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                
                # Small delay for better visualization
                time.sleep(0.02)
            
            success = "Success!" if info.get('is_success', False) else "Failed"
            print(f"Episode {episode}: {success} | Reward: {ep_reward:.2f}")
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    env.close()


def record_video(args):
    """Record episodes to a video file."""
    
    if not HAS_IMAGEIO:
        print("Error: imageio is required for recording videos")
        print("Install with: pip install imageio imageio-ffmpeg")
        return
    
    print("\n" + "=" * 60)
    print("RECORDING VIDEO")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60 + "\n")
    
    # Create environment with rgb_array rendering for recording
    env = gym.make(args.env_id, render_mode='rgb_array')
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = load_model(args.checkpoint, vec_env)
    print(f"Model loaded from {args.checkpoint}\n")
    
    frames = []
    
    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # Capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        success = "✓" if info.get('is_success', False) else "✗"
        print(f"Episode {episode + 1}/{args.n_episodes}: {success} | Reward: {ep_reward:.2f}")
    
    env.close()
    
    # Save video
    if frames:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(frames)} frames to {args.output}...")
        imageio.mimsave(str(output_path), frames, fps=args.fps)
        print(f"Video saved: {args.output}")
    else:
        print("Warning: No frames captured!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize or record a trained agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--env-id', type=str, default='PandaPickAndPlace-v3',
                        help='Environment ID')
    parser.add_argument('--mode', type=str, default='watch',
                        choices=['watch', 'record'],
                        help='Visualization mode')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='./videos/demo.mp4',
                        help='Output video file (for record mode)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS')
    
    args = parser.parse_args()
    
    if args.mode == 'watch':
        watch_agent(args)
    else:
        record_video(args)


if __name__ == "__main__":
    main()
