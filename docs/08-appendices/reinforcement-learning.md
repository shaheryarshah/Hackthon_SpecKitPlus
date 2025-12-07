# Appendix B: Reinforcement Learning for Robot Control

Reinforcement Learning (RL) has emerged as a powerful paradigm for robot control, enabling robots to learn complex behaviors through interaction with their environment. This appendix covers the fundamentals of RL for robotics, implementation techniques, simulation-to-reality transfer, and practical applications. RL is particularly valuable for robotics because it can handle high-dimensional continuous state and action spaces, and can learn behaviors that are difficult to program explicitly.

## Learning Outcomes

After studying this appendix, you should be able to:
- Understand the fundamentals of reinforcement learning in robotic contexts
- Implement basic RL algorithms for robot control tasks
- Design reward functions for robotic manipulation and navigation
- Apply simulation-to-reality transfer techniques for RL policies
- Evaluate the performance and safety of RL-based control systems
- Understand the limitations and challenges of RL in robotics

## Core Concepts

### Markov Decision Process (MDP) Formulation
Robotic control problems are often formulated as MDPs:
- **States (S)**: Robot configuration, sensor readings, environment state
- **Actions (A)**: Joint commands, velocity commands, or high-level actions
- **Rewards (R)**: Scalar feedback guiding learning (e.g., reaching, avoiding obstacles)
- **Transitions (P)**: State transition probabilities (or deterministic in robotics)

### Continuous Control Challenges
RL in robotics faces unique challenges:
- **Continuous Action Spaces**: Most robotic tasks require continuous control
- **Safety Constraints**: Actions must be safe for robot and environment
- **Sample Efficiency**: Physical robots have limited trial-and-error budget
- **Real-time Requirements**: Control decisions in dynamic environments

### Exploration vs. Exploitation
Critical trade-off in RL:
- **Exploration**: Trying new actions to discover better policies
- **Exploitation**: Using known good actions to maximize rewards
- **Curriculum Learning**: Gradually increasing task complexity

## Equations and Models

### Bellman Equation
The optimal value function in reinforcement learning:

```
V*(s) = max_a Σ_s' P(s'|s, a) [R(s, a, s') + γV*(s')]
```

Where:
- `V*(s)` is the optimal value of state s
- `γ` is the discount factor (0 ≤ γ < 1)
- `P(s'|s, a)` is the probability of transitioning to state s' from state s with action a
- `R(s, a, s')` is the reward for the transition

### Policy Gradient Theorem
For policy optimization in continuous action spaces:

```
∇J(θ) = E[∇log π_θ(a|s) Q_π(s, a)]
```

Where:
- `J(θ)` is the expected return under policy π_θ
- `π_θ(a|s)` is the policy parameterized by θ
- `Q_π(s, a)` is the action-value function

### Deep Deterministic Policy Gradient (DDPG)
Actor-critic algorithm for continuous control:

Actor update:
```
∇_θ^μ J ≈ E[∇_θ^μ μ_θ(s) ∇_a Q(s, a)|s=s_t, a=μ_θ(s)]
```

Critic update:
```
L = E[(r + γQ'(s', μ'(s')) - Q(s, a))²]
```

Where:
- `μ_θ(s)` is the actor (policy) network
- `Q(s, a)` is the critic (value) network
- `Q'` and `μ'` are target networks

## Code Example: Deep Deterministic Policy Gradient for Robot Control

Here's an implementation of DDPG for robotic control:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import gym
from torch.distributions import Normal


# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([torch.FloatTensor(e.state) for e in batch])
        actions = torch.stack([torch.FloatTensor(e.action) for e in batch])
        rewards = torch.stack([torch.FloatTensor([e.reward]) for e in batch])
        next_states = torch.stack([torch.FloatTensor(e.next_state) for e in batch])
        dones = torch.stack([torch.FloatTensor([e.done]) for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for policy"""
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
    
    def forward(self, state):
        """Forward pass to get action"""
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))  # Tanh to bound actions
        return self.max_action * a


class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        """Forward pass to get Q-value"""
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Get first Q-value"""
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1


class DDPGAgent:
    """DDPG agent implementation for robotic control"""
    
    def __init__(self, state_dim, action_dim, max_action, lr=1e-4, tau=0.005, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr = lr
        self.tau = tau  # Target network update rate
        self.gamma = gamma  # Discount factor
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        
        # Noise for exploration
        self.exploration_noise = 0.1
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
    
    def select_action(self, state, add_noise=True):
        """Select action using the current policy"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(self, batch_size=256):
        """Train the agent on a batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return  # Not enough experiences yet
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        
        # Update critic
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic.Q1(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filename):
        """Save the model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, f"{filename}_ddpg.pt")
        print(f"Model saved to {filename}_ddpg.pt")
    
    def load(self, filename):
        """Load the model"""
        checkpoint = torch.load(f"{filename}_ddpg.pt", map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filename}_ddpg.pt")


class RobotEnvironment:
    """Simulated robotic environment for training"""
    
    def __init__(self, task="reach"):
        self.task = task
        self.state_dim = 6  # [x, y, z, target_x, target_y, target_z]
        self.action_dim = 3  # [dx, dy, dz] - relative position change
        self.max_action = 0.1  # Maximum action magnitude
        self.goal_threshold = 0.1  # Distance threshold to consider goal reached
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        # Random robot position
        self.robot_pos = np.random.uniform(-1.0, 1.0, 3)
        
        # Random target position
        self.target_pos = np.random.uniform(-1.0, 1.0, 3)
        
        # Ensure target is not too close initially
        while np.linalg.norm(self.robot_pos - self.target_pos) < 0.5:
            self.target_pos = np.random.uniform(-1.0, 1.0, 3)
        
        state = np.concatenate([self.robot_pos, self.target_pos])
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Apply action (relative movement)
        new_pos = self.robot_pos + action
        
        # Clamp position to workspace
        new_pos = np.clip(new_pos, -2.0, 2.0)
        
        # Update robot position
        self.robot_pos = new_pos
        
        # Calculate distance to target
        dist = np.linalg.norm(self.robot_pos - self.target_pos)
        
        # Compute reward
        reward = -dist  # Negative distance as reward
        
        # Add shaping reward for progress
        prev_dist = np.linalg.norm(self.robot_pos - action - self.target_pos)
        if dist < prev_dist:
            reward += 0.1  # Small bonus for making progress
        
        # Add bonus for reaching target
        if dist < self.goal_threshold:
            reward += 10.0  # Large bonus for reaching target
        
        # Create new state
        state = np.concatenate([self.robot_pos, self.target_pos])
        
        # Check if episode is done
        done = dist < self.goal_threshold
        if done:
            # Reset target to continue training
            self.target_pos = np.random.uniform(-1.0, 1.0, 3)
            state = np.concatenate([self.robot_pos, self.target_pos])
            # Don't set done=True in this case to continue training
        
        # Limit episode length
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        if self.step_count > 1000:  # Max steps
            done = True
            self.step_count = 0
        
        info = {'distance': dist, 'reached_target': dist < self.goal_threshold}
        
        return state, reward, done, info


def train_robot_rl_agent(episodes=1000, max_steps=1000):
    """Train a robotic agent using DDPG"""
    # Create environment
    env = RobotEnvironment()
    
    # Create agent
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action
    )
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(max_steps):
            # Select action with noise
            action = agent.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Train agent
            if len(agent.replay_buffer) > 256:
                actor_loss, critic_loss = agent.train()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Distance: {info.get('distance', 0):.3f}")
    
    # Save trained agent
    agent.save("robot_ddpg_agent")
    
    return agent, episode_rewards


def test_robot_agent(agent, episodes=10):
    """Test the trained agent"""
    env = RobotEnvironment()
    
    # Ensure agent is in eval mode
    agent.actor.eval()
    
    success_count = 0
    distances = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        print(f"Testing Episode {episode + 1}")
        
        for step in range(1000):  # Max steps for testing
            # Select action without noise
            action = agent.select_action(state, add_noise=False)
            
            # Execute action
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            distance = info['distance']
            distances.append(distance)
            
            if info.get('reached_target', False):
                success_count += 1
                print(f"  Success! Reached target in {step} steps. Distance: {distance:.3f}")
                break
            
            if done:
                break
        
        print(f"  Episode reward: {episode_reward:.2f}")
    
    success_rate = success_count / episodes
    avg_distance = np.mean(distances) if distances else float('inf')
    
    print(f"\nTesting Results:")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{episodes})")
    print(f"Average Distance: {avg_distance:.3f}")
    
    return success_rate, avg_distance


def main():
    """Example usage of the RL for robotics"""
    print("Reinforcement Learning for Robot Control")
    print("=" * 50)
    
    print("Training DDPG agent for robotic control...")
    agent, rewards = train_robot_rl_agent(episodes=1000)
    
    print(f"\nTraining completed!")
    print(f"Last episode reward: {rewards[-1] if rewards else 0:.2f}")
    print(f"Average reward over last 100 episodes: {np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards):.2f}")
    
    print(f"\nTesting trained agent...")
    success_rate, avg_distance = test_robot_agent(agent, episodes=10)
    
    print("\nThis example demonstrates:")
    print("- Implementation of DDPG for continuous robotic control")
    print("- Use of replay buffer for efficient learning")
    print("- Actor-critic architecture for policy and value learning")
    print("- Testing procedure to evaluate learned policies")
    
    print("\nFor real robotic applications, additional considerations include:")
    print("- Safety constraints and emergency stops")
    print("- Transfer from simulation to physical robots")
    print("- Integration with perception systems")
    print("- Handling partial observability")


if __name__ == "__main__":
    main()
```

## Simulation Demonstration

This implementation demonstrates how to apply reinforcement learning to robotic control tasks. The DDPG algorithm is used for continuous control, which is typical for robot applications. The example creates a simulated environment where a robot learns to reach target positions, which could be extended to more complex tasks like manipulation or navigation. The code can be integrated with real robot hardware after proper safety considerations.

## Hands-On Lab: Reinforcement Learning for Robot Control

In this lab, you'll implement and test RL algorithms for robotic control:

1. Implement the DDPG algorithm for continuous control
2. Create a simulated robotic environment
3. Train an agent to perform a robotic task
4. Transfer the learned policy to a physical robot (if available)
5. Evaluate the performance and safety of the RL controller

### Required Equipment:
- Python development environment
- PyTorch or TensorFlow
- (Optional) Robot simulation environment (Gazebo, Isaac Sim)
- (Optional) Physical robot for testing

### Instructions:
1. Create a new Python package for RL experiments
2. Implement the DDPG agent as shown in the example
3. Create a robotic task environment (e.g., reaching, navigation)
4. Train the agent on the task with appropriate reward shaping
5. Test the learned policy in simulation
6. (If available) Test on a physical robot with safety measures
7. Evaluate the performance metrics (success rate, efficiency)
8. Document the training process and results

## Common Pitfalls & Debugging Notes

- **Sample Efficiency**: RL algorithms often require many samples; use simulation for pre-training
- **Reward Shaping**: Poor reward functions can lead to unexpected behaviors
- **Safety Concerns**: Physical robots need safety constraints during learning
- **Hyperparameter Sensitivity**: RL algorithms can be sensitive to hyperparameters
- **Exploration vs. Safety**: Balancing exploration with safety requirements
- **Transfer Issues**: Policies learned in simulation may not transfer to reality
- **Computational Requirements**: RL training can be computationally intensive

## Summary & Key Terms

**Key Terms:**
- **Markov Decision Process (MDP)**: Mathematical framework for RL problems
- **Actor-Critic**: Architecture with separate policy and value networks
- **Experience Replay**: Technique to break correlation in training data
- **Target Networks**: Slowly updated networks for stable training
- **Policy Gradient**: Method for optimizing policy parameters
- **Continuous Control**: RL with continuous action spaces
- **Sample Efficiency**: How many samples needed to learn effective policy

## Further Reading & Citations

1. Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971.
2. Schulman, J., et al. (2015). "Trust region policy optimization." International Conference on Machine Learning.
3. Levine, S., et al. (2016). "End-to-end training of deep visuomotor policies." Journal of Machine Learning Research.
4. Pomerleau, D. A. (1989). "ALVINN: An autonomous land vehicle in a neural network." Advances in Neural Information Processing Systems.

## Assessment Questions

1. Explain how the DDPG algorithm addresses the challenges of continuous control in robotics.
2. What are the key components of a reinforcement learning system for robot control?
3. Describe the role of experience replay in improving sample efficiency.
4. How would you design a reward function for a robotic manipulation task?
5. What safety considerations are essential when applying RL to physical robots?

---
**Previous**: [Hardware Requirements and Lab Options](./hardware.md)  
**Next**: [Appendix C: Sim-to-Real Transfer Techniques](./sim-to-real.md)