from dynamic_snake_env import DynamicSnakeEnv
import random

env = DynamicSnakeEnv()

while True:
    state = env.reset()
    done = False

    while not done:
        action = random.choice([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        state, reward, done = env.step(action)

