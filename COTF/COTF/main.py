import numpy as np
from data_generator import generate_data
from cotf_framework import cotf_framework

def load_data():
    num_tasks = 20000
    num_local_workers = 20000
    num_coop_workers = 20000
    
    tasks = []
    for i in range(num_tasks):
        task = {
            'id': i,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E','F','G','H','I','J'], size=10, replace=False).tolist(),
            'start_time': np.random.uniform(0, 10),
            'waiting_time': np.random.uniform(0.5, 2.0),
            'base_revenue': np.random.uniform(5, 10),
            'uncover_skills': set()
        }

        tasks.append(task)
    
    local_workers = []
    for i in range(num_local_workers):
        worker = {
            'id': i,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E','F','G','H','I','J'], size=2, replace=False).tolist(),
            'available_time': np.random.uniform(0, 10),
            'history_revenue': 3
        }
        local_workers.append(worker)
    
    coop_workers = []
    for i in range(num_coop_workers):
        worker = {
            'id': i + num_local_workers,
            'location': [np.random.uniform(0, 10), np.random.uniform(0, 10)],
            'skills': np.random.choice(['A', 'B', 'C', 'D', 'E','F','G','H','I','J'], size=2, replace=False).tolist(),
            'available_time': np.random.uniform(0, 10),
            'history_revenue': 3
        }
        coop_workers.append(worker)
    
    return tasks, local_workers, coop_workers

def main():
    tasks, local_workers, coop_workers = load_data()
    worker_velocity = 0.1
    
    print("\nRunning COTF Framework with PQROS strategy:")
    finish_tasks, metrics = cotf_framework(
        tasks.copy(), local_workers.copy(), coop_workers.copy(),
        worker_velocity, strategy_type="PQROS"
    )
    print(f"Platform Total Revenue: {metrics.ptr:.2f}")
    
    print("\nRunning COTF Framework with DQO strategy:")
    finish_tasks, metrics = cotf_framework(
        tasks.copy(), local_workers.copy(), coop_workers.copy(),
        worker_velocity, strategy_type="DQO"
    )
    print(f"Platform Total Revenue: {metrics.ptr:.2f}")

if __name__ == "__main__":
    main() 