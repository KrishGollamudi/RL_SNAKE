import torch
import torch.nn as nn
from snake_env import SnakeEnv
import pygame
import time

device = torch.device("cpu")

# 1. Rebuild the exact same brain structure
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(15, 128),  # 15 Inputs matching the Graduate Environment
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 3)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# 2. Load the trained weights
model = ActorCritic().to(device)
checkpoint_file = "snake_checkpoint.pth"

try:
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # ⚠️ CRITICAL: Sets the model to "Test Mode" (turns off training quirks)
    print(f"✅ Loaded AI Brain from Episode {checkpoint['episode']} successfully!")
except FileNotFoundError:
    print("❌ Could not find 'snake_checkpoint.pth'. Make sure you trained the AI first!")
    exit()

# 3. The Play Loop
def play():
    env = SnakeEnv()
    state = env.reset()
    
    print("🎮 AI is now playing the game... Watch it hunt!")
    
    while True:
        # We slow the game down to 10-15 FPS so the professor can actually watch it play
        env.clock.tick(15) 
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 4. The AI makes a decision
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            logits, _ = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            # 🛑 NO MORE RANDOMNESS (ENTROPY)! 
            # We use argmax to pick the action with the absolute highest probability.
            action = torch.argmax(probs).item()

        state, reward, done, _ = env.step(action)

        if done:
            print(f"Game Over! Snake reached length: {len(env.snake)}")
            time.sleep(1.5) # Pause for a second so the audience can see the final state
            state = env.reset()

if __name__ == "__main__":
    play()

