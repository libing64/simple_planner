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
        
        # Calculate angle difference (shortest path)
        angle_diff = (block_angle - self.goal_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Stricter success condition
        if dist_to_goal < 5 and abs(angle_diff) < 0.1:
            return np.array([0, 0])
            
        dir_to_goal = to_goal / (dist_to_goal + 1e-6)
        
        # Adaptive parameters based on distance to goal
        if dist_to_goal < 50:
            # Very Close: Precision Mode
            # If angle is bad, prioritize rotation over pushing
            effective_k_rot = 50.0 # Saturate offset quickly
            effective_push_speed = 80.0 # Enough to move against friction
            effective_approach_dist = self.approach_dist
        elif dist_to_goal < 120:
            # Transition
            effective_k_rot = 30.0 
            effective_push_speed = 120.0
            effective_approach_dist = self.approach_dist
        else:
            # Far: Transport Mode
            effective_k_rot = 10.0 
            effective_push_speed = 200.0 # Fast
            effective_approach_dist = self.approach_dist - 10 # Tighter contact
        
        # 2. Determine Target Push Position (behind the block)
        right_vec = np.array([dir_to_goal[1], -dir_to_goal[0]]) # (dy, -dx)
        
        # Limit lateral offset to ensure we hit the block
        # Block has narrow parts (width 20). Safe offset is ~10-15.
        lateral_offset_mag = np.clip(effective_k_rot * angle_diff, -15, 15)
        
        lateral_offset = -right_vec * lateral_offset_mag
        
        # Target position behind block
        target_push_pos = block_pos - dir_to_goal * effective_approach_dist + lateral_offset
        
        # 3. Navigation / Pathfinding
        to_target = target_push_pos - agent_pos
        dist_to_target = np.linalg.norm(to_target)
        
        # Avoidance logic
        # "In front" check:
        dist_along_axis = np.dot(agent_pos - block_pos, dir_to_goal)
        
        target_vel = np.array([0.0, 0.0])
        
        # Relaxed avoidance: only avoid if we are truly in front (positive) 
        # or side-flanking (close to 0)
        if dist_along_axis > 5:
            # We are in front or inside. Need to go around.
            # Side waypoints
            wp_right = block_pos + right_vec * (self.block_radius + 30)
            wp_left = block_pos - right_vec * (self.block_radius + 30)
            
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
                target_vel = (target_push_pos - agent_pos)
        else:
            # We are behind.
            
            # Special logic for "Jamming":
            # If we are close to target_push_pos but also very close to the block, 
            # and angle error is high, we might be pushing straight when we want to rotate.
            # But lateral_offset should handle this.
            
            if dist_to_target > self.tolerance_dist:
                # Positioning phase
                target_vel = to_target * 5.0 
            else:
                # Pushing phase
                # Decompose to_target into longitudinal and lateral
                long_err = np.dot(to_target, dir_to_goal)
                lat_err_vec = to_target - long_err * dir_to_goal
                
                # Lateral correction (maintain line)
                lat_correction = lat_err_vec * 8.0
                
                # Longitudinal component
                # If we are behind target (long_err > 0), close gap + push.
                # If we are at/inside target (long_err <= 0), maintain push speed.
                # Don't let position controller fight the push!
                if long_err > 0:
                    long_vel = dir_to_goal * (effective_push_speed + long_err * 5.0)
                else:
                    long_vel = dir_to_goal * effective_push_speed
                    
                target_vel = long_vel + lat_correction
                
        # Clip max speed
        speed = np.linalg.norm(target_vel)
        if speed > self.approach_speed:
            target_vel = target_vel / speed * self.approach_speed
            
        # Debug
        # if np.random.rand() < 0.01:
        #    print(f"Dist:{dist_to_goal:.1f} Ang:{angle_diff:.2f} Off:{lateral_offset_mag:.1f} Pushing:{dist_along_axis < -20} TgtVel:{target_vel}")
            
        return target_vel
