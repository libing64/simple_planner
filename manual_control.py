import gymnasium as gym
import numpy as np
import pygame
from pusht_env import PushTEnv
import time

def main():
    # Initialize environment
    env = PushTEnv(render_mode="human")
    obs, _ = env.reset(seed=42)
    
    print("Manual Control Mode")
    print("Use Arrow Keys to move the agent.")
    print("Press 'R' to reset.")
    print("Press 'Q' to quit.")
    
    done = False
    truncated = False
    
    # Initial action
    action = np.array([0.0, 0.0])
    speed = 100.0
    
    while not done and not truncated:
        # Handle Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    print("Reset.")

        # Continuous key check
        keys = pygame.key.get_pressed()
        vx = 0.0
        vy = 0.0
        
        if keys[pygame.K_LEFT]:
            vx = -speed
        if keys[pygame.K_RIGHT]:
            vx = speed
        if keys[pygame.K_UP]:
            vy = -speed
        if keys[pygame.K_DOWN]:
            vy = speed
            
        action = np.array([vx, vy])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Optional: Render delay
        time.sleep(0.016) # ~60fps
        
        if terminated:
            print("Success!")
            # env.reset() # Optional auto-reset

    env.close()

if __name__ == "__main__":
    main()

