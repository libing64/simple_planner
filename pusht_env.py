import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

class PushTEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.width = 512
        self.height = 512
        self.window_size = (self.width, self.height)
        
        # Physics constants
        self.dt = 1.0 / 60.0
        self.damping = 0.9  # Friction simulation
        
        # Environment setup
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=self.width, shape=(2,), dtype=np.float32),
            "block_pos": spaces.Box(low=0, high=self.width, shape=(2,), dtype=np.float32),
            "block_angle": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        # Action: target velocity for the agent (vx, vy)
        self.action_space = spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.draw_options = None
        
        # Goal position (fixed for now, or random)
        self.goal_pos = np.array([self.width // 2, self.height // 2], dtype=np.float32)
        self.goal_angle = 0.0

    def _get_obs(self):
        return {
            "agent_pos": np.array(self.agent_body.position, dtype=np.float32),
            "block_pos": np.array(self.block_body.position, dtype=np.float32),
            "block_angle": np.array([self.block_body.angle], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.space = pymunk.Space()
        self.space.damping = self.damping

        # Create walls
        walls = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.width, 0), 1),
            pymunk.Segment(self.space.static_body, (self.width, 0), (self.width, self.height), 1),
            pymunk.Segment(self.space.static_body, (self.width, self.height), (0, self.height), 1),
            pymunk.Segment(self.space.static_body, (0, self.height), (0, 0), 1)
        ]
        for wall in walls:
            wall.elasticity = 0.5
            wall.friction = 1.0
        self.space.add(*walls)

        # Create T-Block
        mass = 1
        moment = pymunk.moment_for_box(mass, (80, 80)) # Approximate moment
        self.block_body = pymunk.Body(mass, moment)
        
        # Define T-shape using two rectangles relative to body center
        # Horizontal bar
        shape1 = pymunk.Poly.create_box(self.block_body, (80, 20))
        shape1.transform = pymunk.Transform(ty=-10) # Shift up slightly? No, pymunk coords. Let's center it.
        # Vertical bar
        shape2 = pymunk.Poly.create_box(self.block_body, (20, 60))
        shape2.transform = pymunk.Transform(ty=20) # Shift down
        
        # More precise T-shape centering
        # H-bar: 80x20. V-bar: 20x60. Total height 80.
        # Center of H-bar relative to top: 10.
        # Center of V-bar relative to top: 20 + 30 = 50.
        # Let's say (0,0) is the center of bounding box 80x80.
        # H-bar center: (0, -30) (if Y points down? Pymunk Y points up usually, but let's check pygame_util)
        # Standard pymunk: Y up. Pygame: Y down. pygame_util handles conversion.
        # Let's just create shapes relative to body center (0,0).
        
        h_bar = pymunk.Poly.create_box(self.block_body, (80, 20))
        # h_bar position: top part of T
        v_bar = pymunk.Poly.create_box(self.block_body, (20, 60))
        # v_bar position: bottom stem
        
        # We need to offset them so they form a T
        # Let's say body position is the "center of mass" or geometric center.
        # Let's make the T fit in 80x80 box.
        # H-bar at top: y range [30, 50] (relative to center 0) -> center y=40?
        # Let's offset vertices manually or use Transform.
        # H-bar: center at (0, 30) (size 80x20 covers y=20 to 40)
        # V-bar: center at (0, -10) (size 20x60 covers y=-40 to 20)
        
        t1 = pymunk.Transform(ty=-30)
        t2 = pymunk.Transform(ty=10)
        
        h_bar = pymunk.Poly(self.block_body, [(-40, -10), (40, -10), (40, 10), (-40, 10)], transform=t1)
        v_bar = pymunk.Poly(self.block_body, [(-10, -30), (10, -30), (10, 30), (-10, 30)], transform=t2)
        
        h_bar.friction = 0.5
        v_bar.friction = 0.5
        h_bar.color = (100, 100, 255, 255)
        v_bar.color = (100, 100, 255, 255)
        
        self.space.add(self.block_body, h_bar, v_bar)
        
        # Initial position for block
        self.block_body.position = (self.width / 2, self.height / 2)
        self.block_body.angle = np.random.uniform(0, 2*np.pi) if seed else 0.0

        # Create Agent (Pusher)
        self.agent_radius = 15
        agent_mass = 1
        agent_moment = pymunk.moment_for_circle(agent_mass, 0, self.agent_radius)
        self.agent_body = pymunk.Body(agent_mass, agent_moment)
        self.agent_body.position = (100, 100) # Start corner
        self.agent_shape = pymunk.Circle(self.agent_body, self.agent_radius)
        self.agent_shape.friction = 0.5
        self.agent_shape.color = (255, 100, 100, 255)
        self.space.add(self.agent_body, self.agent_shape)
        
        # Connect agent to a control body for movement (optional, or just set velocity)
        # Setting velocity directly is easier for "velocity control"
        
        # Goal (visual only for physics, but needed for reward)
        # Fixed goal for now: Center with angle 0? Or random?
        # Let's set a specific goal area
        self.goal_pos = np.array([300, 300], dtype=np.float32)
        self.goal_angle = np.pi / 4

        if self.render_mode == "human":
            self._init_render()

        return self._get_obs(), {}

    def step(self, action):
        # Action is velocity (vx, vy)
        # Clip action
        vx, vy = np.clip(action, -100, 100)
        
        # Apply velocity to agent
        self.agent_body.velocity = (vx * 5, vy * 5) # Scale up a bit
        
        # Step physics
        # Multiple sub-steps for stability
        for _ in range(10):
            self.space.step(self.dt / 10)
            
        # Compute reward
        # Distance to goal position
        block_pos = np.array(self.block_body.position)
        dist = np.linalg.norm(block_pos - self.goal_pos)
        
        # Angle difference
        angle_diff = (self.block_body.angle - self.goal_angle + np.pi) % (2 * np.pi) - np.pi
        angle_dist = np.abs(angle_diff)
        
        reward = -dist - 50 * angle_dist # Simple dense reward
        
        terminated = False
        truncated = False
        
        # Check success
        if dist < 20 and angle_dist < 0.2:
            reward += 1000
            terminated = True
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            # Pymunk Y is up, Pygame Y is down. Pymunk debug draw handles this if configured?
            # Or we just configure space gravity? No gravity in top-down.
            # We need to handle coordinates. pymunk.pygame_util defaults to y-up logic if we tell it?
            # Actually standard pymunk is Cartesian (y up). Pygame is Image (y down).
            # DrawOptions usually handles the flip if we set `positive_y_is_up=True` (default) but we need to map it to screen.
            # Let's check typical usage. usually people just accept y-down for pymunk if doing simple stuff, or flip.
            # Let's stick to standard Pymunk (y up) and let DrawOptions flip it.
            pymunk.pygame_util.positive_y_is_up = False # Let's try matching Pygame coords to avoid confusion

    def render(self):
        if self.screen is None:
            self._init_render()
            
        self.screen.fill((255, 255, 255))
        
        # Draw Goal (Green T-block)
        self._draw_block(self.goal_pos, self.goal_angle, (100, 255, 100))
        
        # Draw Physics
        self.space.debug_draw(self.draw_options)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_block(self, pos, angle, color):
        # Define vertices relative to center (matches definitions in reset)
        # H-bar: [-40, -40] to [40, -20]
        h_verts = [(-40, -40), (40, -40), (40, -20), (-40, -20)]
        # V-bar: [-10, -20] to [10, 40]
        v_verts = [(-10, -20), (10, -20), (10, 40), (-10, 40)]
        
        # Transform and draw
        for verts in [h_verts, v_verts]:
            transformed_verts = []
            for x, y in verts:
                # Rotate (standard 2D rotation matrix)
                rx = x * np.cos(angle) - y * np.sin(angle)
                ry = x * np.sin(angle) + y * np.cos(angle)
                # Translate
                tx = rx + pos[0]
                ty = ry + pos[1]
                transformed_verts.append((tx, ty))
            
            pygame.draw.polygon(self.screen, color, transformed_verts)


    def close(self):
        if self.screen is not None:
            pygame.quit()

