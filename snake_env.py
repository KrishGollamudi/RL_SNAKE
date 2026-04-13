import pygame
import random
import numpy as np

class SnakeEnv:
    def __init__(self):
        self.grid_size = 8
        self.block = 50 
        self.width = self.grid_size * self.block
        self.height = self.grid_size * self.block

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Snake (Level 2: Breadcrumbs)")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake = [[4, 4]]
        self.direction = random.choice([(0,1),(1,0),(0,-1),(-1,0)])
        self.frame_iteration = 0  

        self.spawn_food()
        self.spawn_obstacles()
        self.spawn_poison()
        
        # ✅ Track the closest distance to the food
        self.closest_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if self.food not in self.snake and not hasattr(self, 'obstacles') or self.food not in getattr(self, 'obstacles', []):
                break

    def spawn_obstacles(self):
        self.obstacles = []
        for _ in range(1): # ✅ Reduced to 1 obstacle so it isn't an impossible maze yet
            while True:
                obs = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
                if obs not in self.snake and obs != self.food:
                    self.obstacles.append(obs)
                    break

    def spawn_poison(self):
        while True:
            self.poison = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if self.poison not in self.snake and self.poison != self.food and self.poison not in self.obstacles:
                break

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        if pt[0] < 0 or pt[0] >= self.grid_size or pt[1] < 0 or pt[1] >= self.grid_size:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles or pt == self.poison:
            return True
        return False

    def step(self, action):
        self.frame_iteration += 1

        dirs = [(0,1), (1,0), (0,-1), (-1,0)]
        idx = dirs.index(self.direction)

        if action == 1: 
            idx = (idx + 1) % 4
        elif action == 2: 
            idx = (idx - 1) % 4

        self.direction = dirs[idx]
        head = [self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1]]

        reward = -0.01  # Ticking clock

        if self.is_collision(head):
            return self.get_state(), -1.0, True, {}

        if self.frame_iteration > 60 * len(self.snake):
            return self.get_state(), -1.0, True, {}

        # --- ✅ THE BREADCRUMBS ---
        current_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        if current_dist < self.closest_dist:
            reward += 0.1  # Reward for breaking the distance record
            self.closest_dist = current_dist
        # --------------------------

        self.snake.insert(0, head)

        if head == self.food:
            reward += 1.0  
            self.spawn_food()
            # Reset the distance tracker for the new food
            self.closest_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        else:
            self.snake.pop()

        return self.get_state(), reward, False, {}

    def get_state(self):
        head_x, head_y = self.snake[0]

        pt_l = [head_x - 1, head_y]
        pt_r = [head_x + 1, head_y]
        pt_u = [head_x, head_y - 1]
        pt_d = [head_x, head_y + 1]

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        danger_straight = (dir_r and self.is_collision(pt_r)) or (dir_l and self.is_collision(pt_l)) or (dir_u and self.is_collision(pt_u)) or (dir_d and self.is_collision(pt_d))
        danger_right = (dir_u and self.is_collision(pt_r)) or (dir_d and self.is_collision(pt_l)) or (dir_l and self.is_collision(pt_u)) or (dir_r and self.is_collision(pt_d))
        danger_left = (dir_d and self.is_collision(pt_r)) or (dir_u and self.is_collision(pt_l)) or (dir_r and self.is_collision(pt_u)) or (dir_l and self.is_collision(pt_d))

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            self.food[0] < head_x,  
            self.food[0] > head_x,  
            self.food[1] < head_y,  
            self.food[1] > head_y,
            self.poison[0] < head_x, 
            self.poison[0] > head_x, 
            self.poison[1] < head_y, 
            self.poison[1] > head_y  
        ]

        return np.array(state, dtype=np.float32)

    def render(self):
        self.screen.fill((0,0,0))
        for s in self.snake:
            pygame.draw.rect(self.screen, (0,255,0), (s[0]*self.block, s[1]*self.block, self.block, self.block))
        pygame.draw.rect(self.screen, (255,0,0), (self.food[0]*self.block, self.food[1]*self.block, self.block, self.block))
        pygame.draw.rect(self.screen, (160,32,240), (self.poison[0]*self.block, self.poison[1]*self.block, self.block, self.block))
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (120,120,120), (obs[0]*self.block, obs[1]*self.block, self.block, self.block))
            
        pygame.display.flip()
        self.clock.tick(60)
