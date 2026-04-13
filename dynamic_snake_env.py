import pygame
import random
import numpy as np

pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
SPEED = 15

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


class DynamicSnakeEnv:

    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dynamic Snake RL")
        self.clock = pygame.time.Clock()
        self.reset()

    # ================= RESET =================
    def reset(self):
        self.direction = "RIGHT"

        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [
            self.head[:],
            [self.head[0] - BLOCK_SIZE, self.head[1]],
            [self.head[0] - 2 * BLOCK_SIZE, self.head[1]]
        ]

        self.score = 0
        self.frame_iteration = 0

        self.food = self._place_object()
        self.poison = self._place_object()

        # ✅ Controlled dynamic obstacles (with velocity)
        self.obstacles = []
        for _ in range(3):
            pos = self._place_object()
            velocity = random.choice([
                (BLOCK_SIZE, 0),
                (-BLOCK_SIZE, 0),
                (0, BLOCK_SIZE),
                (0, -BLOCK_SIZE)
            ])
            self.obstacles.append({
                "pos": pos,
                "vel": velocity
            })

        return self._get_state()

    # ================= OBJECT PLACEMENT =================
    def _place_object(self):
        while True:
            x = random.randrange(0, WIDTH, BLOCK_SIZE)
            y = random.randrange(0, HEIGHT, BLOCK_SIZE)
            point = [x, y]

            if point not in self.snake:
                return point

    # ================= STEP =================
    def step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move snake
        self._move(action)
        self.snake.insert(0, self.head[:])

        # Move obstacles smoothly
        self._move_obstacles()

        reward = 0
        done = False

        # Collision
        if self._is_collision():
            return self._get_state(), -10, True

        # Food
        if self.head == self.food:
            self.score += 1
            reward += 10
            self.food = self._place_object()

        # Poison
        elif self.head == self.poison:
            reward -= 5
            self.poison = self._place_object()
            self.snake.pop()

        else:
            self.snake.pop()

        # Anti-loop constraint
        if self.frame_iteration > 100 * len(self.snake):
            return self._get_state(), -10, True

        reward -= 0.1

        self._update_ui()
        self.clock.tick(SPEED)

        return self._get_state(), reward, done

    # ================= OBSTACLE MOVEMENT =================
    def _move_obstacles(self):
        for obs in self.obstacles:
            x, y = obs["pos"]
            vx, vy = obs["vel"]

            new_x = x + vx
            new_y = y + vy

            # Bounce from walls
            if new_x < 0 or new_x >= WIDTH:
                vx = -vx
                new_x = x + vx

            if new_y < 0 or new_y >= HEIGHT:
                vy = -vy
                new_y = y + vy

            obs["pos"] = [new_x, new_y]
            obs["vel"] = (vx, vy)

    # ================= COLLISION =================
    def _is_collision(self):
        return self._is_collision_at(self.head)

    def _is_collision_at(self, point):
        x, y = point

        # Wall
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True

        # Self
        if point in self.snake[1:]:
            return True

        # Obstacles
        for obs in self.obstacles:
            if point == obs["pos"]:
                return True

        return False

    # ================= RENDER =================
    def _update_ui(self):
        self.display.fill(BLACK)

        # Snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN,
                             pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))

        # Food
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        # Poison
        pygame.draw.rect(self.display, WHITE,
                         pygame.Rect(self.poison[0], self.poison[1], BLOCK_SIZE, BLOCK_SIZE))

        # Obstacles
        for obs in self.obstacles:
            x, y = obs["pos"]
            pygame.draw.rect(self.display, BLUE,
                             pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

    # ================= MOVEMENT =================
    def _move(self, action):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head

        if self.direction == "RIGHT":
            x += BLOCK_SIZE
        elif self.direction == "LEFT":
            x -= BLOCK_SIZE
        elif self.direction == "UP":
            y -= BLOCK_SIZE
        elif self.direction == "DOWN":
            y += BLOCK_SIZE

        self.head = [x, y]

    # ================= STATE =================
    def _get_state(self):
        head_x, head_y = self.head

        # Relative positions
        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y

        poison_dx = self.poison[0] - head_x
        poison_dy = self.poison[1] - head_y

        # Nearby points
        point_l = [head_x - BLOCK_SIZE, head_y]
        point_r = [head_x + BLOCK_SIZE, head_y]
        point_u = [head_x, head_y - BLOCK_SIZE]
        point_d = [head_x, head_y + BLOCK_SIZE]

        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        state = [
            # Danger straight
            (dir_r and self._is_collision_at(point_r)) or
            (dir_l and self._is_collision_at(point_l)) or
            (dir_u and self._is_collision_at(point_u)) or
            (dir_d and self._is_collision_at(point_d)),

            # Danger right
            (dir_u and self._is_collision_at(point_r)) or
            (dir_d and self._is_collision_at(point_l)) or
            (dir_l and self._is_collision_at(point_u)) or
            (dir_r and self._is_collision_at(point_d)),

            # Danger left
            (dir_d and self._is_collision_at(point_r)) or
            (dir_u and self._is_collision_at(point_l)) or
            (dir_r and self._is_collision_at(point_u)) or
            (dir_l and self._is_collision_at(point_d)),

            # Direction
            dir_l, dir_r, dir_u, dir_d,

            # Food direction
            food_dx < 0,
            food_dx > 0,
            food_dy < 0,
            food_dy > 0,

            # Poison direction
            poison_dx < 0,
            poison_dx > 0,
            poison_dy < 0,
            poison_dy > 0
        ]

        return np.array(state, dtype=int)
