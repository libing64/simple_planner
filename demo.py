import gymnasium as gym
import numpy as np
import pygame
from pusht_env import PushTEnv
from planner import SimplePlanner
import time

def main():
    # Initialize environment with render mode for visualization
    env = PushTEnv(render_mode="human")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Initialize planner
    # Goal is fixed in env for now: (300, 300)
    goal_pos = env.goal_pos
    planner = SimplePlanner(goal_pos=goal_pos)
    
    print("Starting simulation...")
    print(f"Goal Position: {goal_pos}")
    
    done = False
    truncated = False
    step = 0
    max_steps = 1000
    
    while not done and not truncated and step < max_steps:
        # Get action from planner
        action = planner.get_action(obs)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Optional: Slow down rendering
        time.sleep(0.02)
        
        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        step += 1
        if step % 100 == 0:
            dist = np.linalg.norm(obs['block_pos'] - goal_pos)
            print(f"Step {step}: Distance to goal = {dist:.2f}")

    if done:
        print("Success! Target reached.")
    else:
        print("Time limit reached.")
        
    env.close()

if __name__ == "__main__":
    main()

