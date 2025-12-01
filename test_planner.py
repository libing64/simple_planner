import gymnasium as gym
import numpy as np
from pusht_env import PushTEnv
from planner import SimplePlanner

def test_planner():
    env = PushTEnv(render_mode=None)
    
    seeds = [42, 100, 2023, 5, 99]
    success_count = 0
    total_steps = 0
    
    print(f"Testing planner on {len(seeds)} episodes...")
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        goal_pos = env.goal_pos
        goal_angle = env.goal_angle
        planner = SimplePlanner(goal_pos=goal_pos, goal_angle=goal_angle)
        
        done = False
        truncated = False
        steps = 0
        max_steps = 1500
        
        while not done and steps < max_steps:
            action = planner.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            steps += 1
            
        final_dist = np.linalg.norm(obs['block_pos'] - goal_pos)
        final_angle = obs['block_angle'][0]
        
        status = "Success" if done else "Failed"
        print(f"Seed {seed}: {status} in {steps} steps. Dist: {final_dist:.2f}, Angle: {final_angle:.2f}")
        
        if done:
            success_count += 1
            total_steps += steps
            
    success_rate = success_count / len(seeds)
    avg_steps = total_steps / success_count if success_count > 0 else 0
    
    print(f"\nResults:")
    print(f"Success Rate: {success_rate * 100:.1f}%")
    print(f"Avg Steps (Success): {avg_steps:.1f}")

if __name__ == "__main__":
    test_planner()

