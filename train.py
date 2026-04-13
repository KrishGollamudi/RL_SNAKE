import torch
import torch.nn as nn
import torch.optim as optim
from snake_env import SnakeEnv
import pygame
import os # ✅ NEW: Needed for checking if the checkpoint file exists

device = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(15, 128),  # 15 Inputs for the "Graduate Level" Environment
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 3)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma = 0.99
lam = 0.95
clip = 0.2
entropy_coef = 0.10  # High curiosity to break loops

class Memory:
    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def clear(self):
        self.__init__()

memory = Memory()

# --- ✅ NEW: CHECKPOINT FUNCTIONS ---
CHECKPOINT_FILE = "snake_checkpoint.pth"

def save_checkpoint(episode):
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"💾 Checkpoint saved at Episode {episode}!")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        print(f"🔄 Found checkpoint '{CHECKPOINT_FILE}'. Loading...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f"✅ Successfully resumed from Episode {start_episode}!")
        return start_episode
    else:
        print("🆕 No checkpoint found. Starting fresh brain!")
        return 0
# ------------------------------------

def select_action(state):
    state = torch.FloatTensor(state).to(device)
    logits, value = model(state)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, 1e-6, 1.0)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()

    memory.states.append(state)
    memory.actions.append(action)
    memory.log_probs.append(dist.log_prob(action))
    memory.values.append(value.squeeze())

    return action.item()

def compute_gae(next_value):
    rewards = memory.rewards
    values = memory.values + [next_value]
    dones = memory.dones
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        delta = rewards[i] + gamma * values[i+1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        returns.insert(0, gae + values[i])
    return returns

def update():
    if len(memory.rewards) == 0:
        return
    next_value = torch.tensor(0.0).to(device)
    returns = compute_gae(next_value)

    states = torch.stack(memory.states).detach()
    actions = torch.stack(memory.actions).detach()
    old_log_probs = torch.stack(memory.log_probs).detach()
    returns = torch.stack(returns).detach()
    values = torch.stack(memory.values).detach()

    advantages = returns - values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = torch.zeros_like(advantages)

    for _ in range(4):
        logits, new_values = model(states)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, 1e-6, 1.0)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - new_values.squeeze()).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - (entropy_coef * entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    memory.clear()

if __name__ == "__main__":
    env = SnakeEnv()

    print("--- RL Snake Training Started ---")
    
    # ✅ NEW: Load the checkpoint if it exists
    start_episode = load_checkpoint()

    for episode in range(start_episode, 10000): # Raised ceiling to 10,000
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Comment out env.render() if you want it to train silently in the background
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Save just in case you close out manually with the 'X' button
                    save_checkpoint(episode) 
                    pygame.quit()
                    quit()

            action = select_action(state)
            next_state, reward, done, _ = env.step(action)

            memory.rewards.append(torch.tensor(reward, dtype=torch.float32))
            memory.dones.append(torch.tensor(float(done)))

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        update()
        
        # ✅ NEW: Save automatically every 100 episodes
        if episode > 0 and episode % 100 == 0:
            save_checkpoint(episode)
        
        # Print every 10 episodes to keep terminal clean
        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {steps} | Snake Length: {len(env.snake)}")
