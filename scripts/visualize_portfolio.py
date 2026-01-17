#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pathlib import Path

import gymnasium as gym
import panda_gym

from stable_baselines3 import SAC

try:
    from sb3_contrib import TQC
    HAS_TQC = True
except ImportError:
    HAS_TQC = False

try:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def load_model(checkpoint_path, env, device='auto'):
    """Load model from checkpoint."""
    checkpoint_path = str(checkpoint_path)
    if HAS_TQC:
        try:
            return TQC.load(checkpoint_path, env=env, device=device)
        except:
            pass
    return SAC.load(checkpoint_path, env=env, device=device)


def add_text_overlay(frame, text, position='top', font_size=24, color=(255, 255, 255)):
    """Add text overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == 'top':
        x = (frame.shape[1] - text_width) // 2
        y = 20
    elif position == 'bottom':
        x = (frame.shape[1] - text_width) // 2
        y = frame.shape[0] - text_height - 20
    elif position == 'top-left':
        x, y = 20, 20
    elif position == 'top-right':
        x = frame.shape[1] - text_width - 20
        y = 20
    else:
        x, y = position
    
    # Draw shadow for better visibility
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)
    
    return np.array(img)


def add_success_overlay(frame, success):
    """Add success/failure indicator."""
    if success:
        text = "✓ SUCCESS"
        color = (0, 255, 0)  # Green
    else:
        text = "✗ FAILED"
        color = (255, 0, 0)  # Red
    
    return add_text_overlay(frame, text, position='bottom', font_size=32, color=color)


def record_portfolio_video(args):
    """Record polished portfolio demo video."""
    
    if not HAS_DEPS:
        print("Error: imageio and Pillow required")
        print("Install with: pip install imageio imageio-ffmpeg Pillow")
        return
    
    print("\n" + "="*60)
    print("RECORDING PORTFOLIO DEMO")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Pause frames after success: {args.pause_frames}")
    print("="*60 + "\n")
    
    # Create environment
    env = gym.make(args.env_id, render_mode='rgb_array')
    
    # Load model
    model = load_model(args.checkpoint, env, device=args.device)
    print(f"Model loaded from {args.checkpoint}\n")
    
    frames = []
    total_successes = 0
    
    for episode in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_frames = []
        step_count = 0
        
        while not done:
            # Render frame
            frame = env.render()
            if frame is not None:
                # Add episode counter
                frame = add_text_overlay(
                    frame, 
                    f"Episode {episode + 1}/{args.n_episodes}", 
                    position='top-left',
                    font_size=20
                )
                episode_frames.append(frame)
            
            # Get action and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Check success
        success = info.get('is_success', False)
        if success:
            total_successes += 1
        
        # Add success overlay to last few frames
        if episode_frames:
            # Add success indicator to last N frames
            for i in range(min(args.indicator_frames, len(episode_frames))):
                idx = -(i + 1)
                episode_frames[idx] = add_success_overlay(episode_frames[idx], success)
        
        # Add pause frames after episode completion
        if episode_frames and args.pause_frames > 0:
            last_frame = episode_frames[-1].copy()
            # Add "COMPLETE" text during pause
            last_frame = add_text_overlay(
                last_frame,
                f"Reward: {episode_reward:.1f}",
                position='top-right',
                font_size=18,
                color=(255, 255, 0)
            )
            for _ in range(args.pause_frames):
                episode_frames.append(last_frame)
        
        frames.extend(episode_frames)
        
        status = "✓" if success else "✗"
        print(f"Episode {episode + 1}/{args.n_episodes}: {status} | "
              f"Reward: {episode_reward:.2f} | Steps: {step_count}")
        
        # Add transition frames (fade or blank)
        if args.transition_frames > 0 and episode < args.n_episodes - 1:
            if episode_frames:
                # Simple fade to black transition
                last_frame = episode_frames[-1]
                for i in range(args.transition_frames):
                    alpha = 1.0 - (i / args.transition_frames)
                    faded = (last_frame * alpha).astype(np.uint8)
                    frames.append(faded)
    
    env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_successes}/{args.n_episodes} successful "
          f"({100*total_successes/args.n_episodes:.1f}%)")
    print(f"{'='*60}\n")
    
    # Save video
    if frames:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(frames)} frames to {args.output}...")
        imageio.mimsave(
            str(output_path),
            frames,
            fps=args.fps,
        )
        print(f"Video saved: {args.output}")
        print(f"Duration: {len(frames)/args.fps:.1f} seconds")
    else:
        print("Warning: No frames captured!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record polished portfolio demo videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--env-id', type=str, default='PandaPickAndPlace-v3',
                        help='Environment ID')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='./videos/portfolio_demo.mp4',
                        help='Output video file')
    parser.add_argument('--fps', type=int, default=15,
                        help='Video FPS (lower = slower playback)')
    parser.add_argument('--pause-frames', type=int, default=30,
                        help='Frames to pause after each episode (at fps=15, 30 frames = 2 sec)')
    parser.add_argument('--transition-frames', type=int, default=10,
                        help='Frames for fade transition between episodes')
    parser.add_argument('--indicator-frames', type=int, default=15,
                        help='How many end frames to show success indicator on')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_portfolio_video(args)
