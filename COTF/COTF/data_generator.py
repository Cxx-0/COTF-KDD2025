import numpy as np
import random

def generate_data(num_tasks=10000, num_local_workers=120000, num_coop_workers=120000, worker_velocity=0.1, seed=42):
    # 固定随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    tasks = []
    for i in range(num_tasks):
        task = {
            'id': i,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=3, replace=False).tolist(),
            'start_time': np.random.uniform(0, 10),
            'waiting_time': np.random.uniform(0.5, 2.0),
            'base_revenue': np.random.uniform(50, 100),
            'uncover_skills': set()
        }
        tasks.append(task)
    
    local_workers = []
    for i in range(num_local_workers):
        worker = {
            'id': i,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=2, replace=False).tolist(),
            'available_time': 0,
            'history_revenue': 30,
            'assigned': False
        }
        local_workers.append(worker)
    
    coop_workers = []
    for i in range(num_coop_workers):
        worker = {
            'id': i + num_local_workers,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=2, replace=False).tolist(),
            'available_time': 0,
            'history_revenue': 30,
            'assigned': False
        }
        coop_workers.append(worker)
    
    return tasks, local_workers, coop_workers 