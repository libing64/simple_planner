import numpy as np

class SimplePlanner:
    def __init__(self, goal_pos, goal_angle=0):
        self.goal_pos = goal_pos
        self.goal_angle = goal_angle
        self.state = "APPROACH" # APPROACH, PUSH, BACKOFF
        self.approach_offset = 120 # Distance behind block to line up
        self.push_threshold = 100 # How close to goal before backing off (if needed)
        
    def get_action(self, obs):
        agent_pos = obs['agent_pos']
        block_pos = obs['block_pos']
        block_angle = obs['block_angle'][0]
        
        # Vector from block to goal
        to_goal = self.goal_pos - block_pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal < 10:
            return np.array([0, 0]) # Close enough
            
        dir_to_goal = to_goal / (dist_to_goal + 1e-6)
        
        # Target position for agent to start pushing (behind the block)
        push_start_pos = block_pos - dir_to_goal * self.approach_offset
        
        # Vector from agent to push start
        to_start = push_start_pos - agent_pos
        dist_to_start = np.linalg.norm(to_start)
        
        # Vector from agent to block
        to_block = block_pos - agent_pos
        dist_to_block = np.linalg.norm(to_block)
        
        # Simple State Machine
        
        # 1. If we are far from the pushing line/position, go there first
        # We want to be behind the block.
        # Check if we are "in front" of the block relative to goal?
        # Dot product: (agent - block) . dir_to_goal > 0 means we are in front.
        
        is_in_front = np.dot(agent_pos - block_pos, dir_to_goal) > -20
        
        Kp = 5.0 # Proportional gain
        
        if is_in_front or dist_to_start > 20:
            # Navigate to push_start_pos
            # Simple P-controller
            
            # Avoid hitting the block if we are on the wrong side?
            # If we are in front, we should go around.
            # Very simple avoidance: go perpendicular first?
            # For this simple planner, let's just go to start pos. 
            # If we plow through the block, physics might be messy but it might eventually work.
            # To be slightly smarter: if blocked, move sideways.
            
            target_vel = to_start * Kp
        else:
            # We are behind and close to start pos. Push!
            # Target is the goal (or through the block)
            target_vel = dir_to_goal * 100 # Max speed push
            
        # Clip velocity
        speed = np.linalg.norm(target_vel)
        if speed > 100:
            target_vel = target_vel / speed * 100
            
        return target_vel

