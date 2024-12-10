from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()
    
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam) # Probability Distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob)
        return action.item()
    
def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # Returns
    future_ret = 0.0
    
    # Compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma*future_ret
        rets[t] = future_ret
    
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs) # joins (concatenates) a sequence of tensors (two or more tensors) along a new dimension
    loss = -log_probs * rets # Gradient term; -ve for maximizing
    loss = torch.sum(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def main():
    env = gym.make('CartPole-v1', render_mode = 'human')
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    
    for epi in range(300):
        state, _ = env.reset()
        print(state)
        for t in range(200): # Cartpole max timestamp is 200
            action = pi.act(state)
            state, reward, done, _, _1 = env.step(action) # Run one timestamp of environment's dynamics
            pi.rewards.append(reward)
            env.render() # Compute the render frames
            if done:
                break
        
        loss = train(pi, optimizer) # Train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset() # onpolicy: Clear memory after training
        print(f'Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}')

if __name__ == '__main__':
    main()