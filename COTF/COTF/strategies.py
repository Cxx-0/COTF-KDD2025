import random
import numpy as np
from utils import can_reach, calculate_max_query_range, calculate_intersection_area, calculate_expected_revenue
from dqo import DQOAgent

def calculate_intersection_area_ratio(task1, task2, current_time, worker_velocity):

    task1_range = calculate_max_query_range(task1, current_time, worker_velocity)
    task2_range = calculate_max_query_range(task2, current_time, worker_velocity)
    
    intersection_area = calculate_intersection_area(
        task1_range, task2_range,
        task1['location'], task2['location']
    )
    task1_area = np.pi * task1_range ** 2
    
    return intersection_area / task1_area if task1_area > 0 else 0

class QueryStrategy:
    def __init__(self, strategy_type="PQROS"):
        self.strategy_type = strategy_type
        self.tasks_history = []
        
        if strategy_type == "DQO":
            state_dim = 5
            action_dim = 7
            self.dqo_agent = DQOAgent(state_dim, action_dim)

    def calculate_query_threshold(self, task, current_time, available_workers_count):
        avg_wait_time = np.mean([t['waiting_time'] for t in self.tasks_history]) if self.tasks_history else task['waiting_time']
        base_threshold = max(0.5, avg_wait_time / (1 + np.log(1 + available_workers_count)))

        if self.strategy_type == "DQO":
            state = self.dqo_agent.get_state(
                current_time,
                task['location'],
                available_workers_count,
                len([t for t in self.tasks_history if t['waiting_time'] > current_time])
            )
            action = self.dqo_agent.select_action(state)
            adjustment_ratio = (action - 3) * 0.1
            return base_threshold * (1 + adjustment_ratio)
        else:
            return base_threshold

    def find_overlapping_tasks(self, task, current_time, worker_velocity):
        overlapping_tasks = []
        
        for old_task in self.tasks_history:
            if old_task['waiting_time'] > current_time and old_task['uncover_skills']:
                if calculate_intersection_area_ratio(task, old_task, current_time, worker_velocity) > 0:
                    overlapping_tasks.append(old_task)
        
        return overlapping_tasks

def greedy_assignment(task, available_workers, current_time):
    selected_workers = []
    remaining_skills = set(task['skills'])
    
    for worker in available_workers:
        if not remaining_skills:
            break
            
        worker_skills = set(worker['skills'])
        matching_skills = worker_skills & remaining_skills
        
        if matching_skills:
            selected_workers.append(worker)
            remaining_skills -= matching_skills
    
    return selected_workers

def random_cooperation_strategy(task, coop_workers, current_time):

    required_skills = set(task['skills'])
    available_workers = [w for w in coop_workers if can_reach(w, task, current_time)]
    
    if not available_workers:
        return None
    
    selected_workers = random.sample(available_workers, min(len(available_workers), len(required_skills)))
    
    team_skills = set()
    for worker in selected_workers:
        team_skills.update(worker['skills'])
    
    if required_skills.issubset(team_skills):
        return selected_workers
    else:
        return None

def precision_query_range_optimization(task, available_workers, local_team, current_time, query_strategy, metrics, worker_velocity):
    if not task['uncover_skills']:
        return []
    
    max_range = calculate_max_query_range(task, current_time, worker_velocity)
    task_location = task['location']
    
    potential_workers = []
    for worker in available_workers:
        worker_location = worker['location']
        distance = np.sqrt((worker_location[0] - task_location[0])**2 + 
                         (worker_location[1] - task_location[1])**2)
        
        if distance <= max_range:
            potential_workers.append(worker)
    
    metrics.update_query_count()
    
    if not potential_workers:
        return []
    
    selected_workers = []
    remaining_skills = task['uncover_skills'].copy()
    
    for worker in potential_workers:
        if not remaining_skills:
            break
            
        worker_skills = set(worker['skills'])
        matching_skills = worker_skills & remaining_skills
        
        if matching_skills:
            selected_workers.append(worker)
            remaining_skills -= matching_skills
    
    return selected_workers 