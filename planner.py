import numpy as np

class SimplePlanner:
    def __init__(self, goal_pos, goal_angle=0):
        self.goal_pos = goal_pos
        self.goal_angle = goal_angle
        # Parameters
        self.approach_dist = 60    # Distance to stage behind block
        self.push_speed = 150       # Max speed for pushing
        self.approach_speed = 150   # Max speed for approaching
        self.k_rot = 20.0           # Gain for angular correction (lateral offset)
        self.block_radius = 60      # Approximate radius for avoidance
        self.tolerance_dist = 20    # Distance tolerance to switch modes
        
    def get_action(self, obs):
        agent_pos = obs['agent_pos']
        block_pos = obs['block_pos']
        block_angle = obs['block_angle'][0]
        
        # 1. Goal Vector and Angle Error
        to_goal = self.goal_pos - block_pos
        dist_to_goal = np.linalg.norm(to_goal)
        
        if dist_to_goal < 10:
            return np.array([0, 0])
            
        dir_to_goal = to_goal / (dist_to_goal + 1e-6)
        
        # Calculate angle difference (shortest path)
        angle_diff = (block_angle - self.goal_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Adaptive parameters based on distance to goal
        if dist_to_goal < 100:
            # Close to goal: precise maneuvering
            effective_k_rot = self.k_rot # High gain for angle
            effective_push_speed = self.push_speed * 0.5 # Slow down
            effective_approach_dist = self.approach_dist
        else:
            # Far from goal: rough transport
            effective_k_rot = 5.0 # Low gain to prevent spinning
            effective_push_speed = self.push_speed
            effective_approach_dist = self.approach_dist
        
        # 2. Determine Target Push Position (behind the block)
        right_vec = np.array([dir_to_goal[1], -dir_to_goal[0]]) # (dy, -dx)
        lateral_offset_mag = np.clip(effective_k_rot * angle_diff, -40, 40)
        lateral_offset = -right_vec * lateral_offset_mag
        
        # Target position behind block
        target_push_pos = block_pos - dir_to_goal * effective_approach_dist + lateral_offset
        
        # 3. Navigation / Pathfinding
        # Check if we can go straight to target_push_pos
        to_target = target_push_pos - agent_pos
        dist_to_target = np.linalg.norm(to_target)
        
        # Avoidance logic
        # "In front" check:
        dist_along_axis = np.dot(agent_pos - block_pos, dir_to_goal)
        
        target_vel = np.array([0.0, 0.0])
        
        if dist_along_axis > -20:
            # We are in front or inside. Need to go around.
            # Side waypoints: block_pos +/- right_vec * (radius + margin)
            wp_right = block_pos + right_vec * (self.block_radius + 20)
            wp_left = block_pos - right_vec * (self.block_radius + 20)
            
            dist_right = np.linalg.norm(wp_right - agent_pos)
            dist_left = np.linalg.norm(wp_left - agent_pos)
            
            if dist_right < dist_left:
                target_wp = wp_right
            else:
                target_wp = wp_left
                
            # Move to waypoint
            to_wp = target_wp - agent_pos
            if np.linalg.norm(to_wp) > 10:
                target_vel = to_wp / (np.linalg.norm(to_wp) + 1e-6) * self.approach_speed
            else:
                # Close to waypoint, now head to target_push_pos
                target_vel = (target_push_pos - agent_pos)
        else:
            # We are behind.
            # If we are far from target_push_pos, go there.
            if dist_to_target > self.tolerance_dist:
                target_vel = to_target * 5.0 # P-gain for positioning
            else:
                # We are in position. PUSH!
                push_drive = dir_to_goal * effective_push_speed
                correction_drive = to_target * 5.0
                
                target_vel = push_drive + correction_drive
                
        # Clip max speed
        speed = np.linalg.norm(target_vel)
        if speed > self.approach_speed:
            target_vel = target_vel / speed * self.approach_speed
            
        return target_vel
