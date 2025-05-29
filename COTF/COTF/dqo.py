import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_steps = 0
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.task_arrival_rates = {}
        self.skill_revenues = {
            'local': {},
            'coop': {}
        }
        
        self.last_ptr = 0
        self.last_tcc = 0
        self.last_query_count = 0

    def get_state(self, current_time, task_location, available_workers_count, tasks_in_range):
        state = np.array([
            current_time,
            task_location[0],
            task_location[1],
            available_workers_count,
            tasks_in_range
        ])
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def calculate_reward(self, metrics):
        ptr_increment = metrics.ptr - self.last_ptr
        tcc_increment = metrics.tcc - self.last_tcc
        query_increment = metrics.query_counts - self.last_query_count
        
        self.last_ptr = metrics.ptr
        self.last_tcc = metrics.tcc
        self.last_query_count = metrics.query_counts
        
        revenue_reward = 5.0 * ptr_increment
        cost_penalty = 0.1 * tcc_increment
        query_penalty = 0.01 * query_increment
        
        efficiency_bonus = 0
        if ptr_increment > 0:
            if query_increment > 0:
                efficiency_bonus = 2.0 * (ptr_increment / query_increment)
            else:
                efficiency_bonus = 2.0 * ptr_increment
        
        coverage_reward = 1.0 * metrics.get_average_skill_coverage()
        
        ptr_threshold_bonus = 0
        if metrics.ptr > 50000:
            ptr_threshold_bonus = 1000
        
        total_reward = revenue_reward + cost_penalty + query_penalty + efficiency_bonus + coverage_reward + ptr_threshold_bonus
        return total_reward

    def update(self, metrics):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.training_steps += 1
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calculate_worker_idle_value(self, worker, current_time):
        idle_value = 0
        for skill in worker['skills']:
            local_revenue = self.skill_revenues['local'].get(skill, 0)
            local_arrival_rate = self.task_arrival_rates.get(('local', skill), 0)
            
            coop_revenue = self.skill_revenues['coop'].get(skill, 0)
            coop_arrival_rate = self.task_arrival_rates.get(('coop', skill), 0)
            
            skill_value = (local_revenue * local_arrival_rate + 
                         coop_revenue * coop_arrival_rate)
            
            idle_value += skill_value
        return idle_value

    def update_statistics(self, task, platform_type, revenue):
        for skill in task['skills']:
            if platform_type == 'local':
                if skill not in self.skill_revenues['local']:
                    self.skill_revenues['local'][skill] = revenue
                else:
                    old_value = self.skill_revenues['local'][skill]
                    self.skill_revenues['local'][skill] = 0.9 * old_value + 0.1 * revenue
            else:
                if skill not in self.skill_revenues['coop']:
                    self.skill_revenues['coop'][skill] = revenue
                else:
                    old_value = self.skill_revenues['coop'][skill]
                    self.skill_revenues['coop'][skill] = 0.9 * old_value + 0.1 * revenue
        
        for skill in task['skills']:
            arrival_key = (platform_type, skill)
            if arrival_key not in self.task_arrival_rates:
                self.task_arrival_rates[arrival_key] = 1
            else:
                self.task_arrival_rates[arrival_key] += 1 