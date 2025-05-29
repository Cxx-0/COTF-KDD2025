import numpy as np
from utils import (
    update_worker, calculate_real_task_revenue, Update_task,
    calculate_max_query_range, calculate_intersection_area,
    calculate_expected_revenue, get_available_workers
)
from strategies import (
    greedy_assignment,
    precision_query_range_optimization,
    QueryStrategy
)
from metrics import Metrics
import random
import torch
import copy

def vectorized_state_calculation(current_time, task_locations, available_workers_count, remaining_tasks):
    states = np.column_stack([
        np.full_like(task_locations[:, 0], current_time),
        task_locations,
        np.full_like(task_locations[:, 0], available_workers_count),
        np.full_like(task_locations[:, 0], remaining_tasks)
    ])
    return torch.FloatTensor(states)

def binary_search_available_workers(worker_availability, current_time, start_idx):
    left, right = start_idx, len(worker_availability)
    while left < right:
        mid = (left + right) // 2
        if worker_availability[mid][0] <= current_time:
            left = mid + 1
        else:
            right = mid
    return left

def cotf_framework(tasks, org_local_workers, org_coop_workers, worker_velocity, strategy_type="PQROS", num_pretrain_epochs=1):
    current_time = 0
    finish_tasks = []
    
    metrics = Metrics()
    query_strategy = QueryStrategy(strategy_type)
    
    if strategy_type == "DQO":
        print("Starting DQO pre-training...")
        for epoch in range(num_pretrain_epochs):
            print(f"Pre-training epoch {epoch + 1}/{num_pretrain_epochs}")
            
            metrics = Metrics()
            pretrain_tasks = copy.deepcopy(tasks)
            pretrain_tasks.sort(key=lambda x: x['start_time'], reverse=True)
            current_time_pretrain = 0
            
            local_workers = copy.deepcopy(org_local_workers)
            coop_workers = copy.deepcopy(org_coop_workers)
            
            # 重置所有工人的分配状态
            for worker in local_workers + coop_workers:
                worker['assigned'] = False
            
            worker_availability = sorted([(w['available_time'], i) for i, w in enumerate(local_workers)])
            current_available_count = 0
            next_worker_idx = 0
            
            while len(pretrain_tasks) > 0:
                task = pretrain_tasks.pop()
                local_revenue = 0
                coop_revenue = 0
                
                current_available_count = binary_search_available_workers(worker_availability, current_time_pretrain, next_worker_idx)
                
                base_threshold = max(0.01, task['waiting_time'] / (1 + np.log(1 + current_available_count)))
                state = query_strategy.dqo_agent.get_state(
                    current_time_pretrain,
                    task['location'],
                    current_available_count,
                    len(pretrain_tasks)
                )
                action = query_strategy.dqo_agent.select_action(state)
                adjustment_ratio = (action - 3) * 0.1
                time_threshold = base_threshold * (1 + adjustment_ratio)
                time_threshold = np.clip(time_threshold, 0.005, 0.05)
                
                if task['waiting_time'] - current_time_pretrain >= time_threshold:
                    task_start_time = current_time_pretrain
                    available_workers = get_available_workers(local_workers, current_time_pretrain)
                    
                    local_team = greedy_assignment(task, available_workers, current_time_pretrain)
                    task_revenue = calculate_real_task_revenue(task, local_team)
                    local_revenue = task_revenue
                    
                    update_task = Update_task(task, local_team)
                    for worker in local_team:
                        update_worker(worker, task)

                    local_skills = set()
                    for worker in local_team:
                        local_skills.update(worker['skills'])
                    local_coverage = len(local_skills & set(task['skills'])) / len(task['skills'])

                    if local_coverage < 1:  
                        coop_available_workers = get_available_workers(coop_workers, current_time_pretrain)
                        coop_team = precision_query_range_optimization(
                            update_task, coop_available_workers, local_team, 
                            current_time_pretrain, query_strategy, metrics, worker_velocity
                        )
                        
                        if coop_team:
                            team = coop_team + local_team
                            new_revenue = calculate_real_task_revenue(task, team)
                            
                            comp_cost = len(coop_team) * 5
                            comm_cost = len(coop_team) * 2
                            total_cost = comp_cost + comm_cost
                            
                            # 只有当收益增量大于成本时才进行合作
                            revenue_increment = new_revenue - local_revenue
                            if revenue_increment > total_cost:
                                metrics.update_cooperation_cost(comp_cost, comm_cost)
                                coop_revenue = new_revenue
                                
                                for worker in coop_team:
                                    update_worker(worker, task)
                            else:
                                coop_revenue = local_revenue
                    else:
                        coop_revenue = local_revenue
                    
                    task_completion_time = current_time_pretrain + task['waiting_time']
                    metrics.update_response_time(task_start_time, task_completion_time)
                    metrics.update_revenue(coop_revenue)
                    
                    total_skills = len(task['skills'])
                    covered_skills = total_skills - len(task['uncover_skills'])
                    coverage_rate = covered_skills / total_skills
                    metrics.update_skill_coverage(id(task), coverage_rate)
                    
                    task['coop_revenue'] = coop_revenue
                    task['local_revenue'] = local_revenue
                    current_time_pretrain = task['start_time']
                else:
                    current_time_pretrain += 0.01
            
            query_strategy.dqo_agent.update(metrics)
        
        print("Pre-training completed.")
        metrics = Metrics()
    
    tasks.sort(key=lambda x: x['start_time'], reverse=True)
    local_workers = copy.deepcopy(org_local_workers)
    coop_workers = copy.deepcopy(org_coop_workers)
    
    # 重置所有工人的分配状态
    for worker in local_workers + coop_workers:
        worker['assigned'] = False
    
    worker_availability = sorted([(w['available_time'], i) for i, w in enumerate(local_workers)])
    current_available_count = 0
    next_worker_idx = 0
    
    while len(tasks) > 0:
        task = tasks.pop()
        local_revenue = 0
        coop_revenue = 0
        
        if strategy_type == "PQROS":
            time_threshold = 0.5
        else:
            current_available_count = binary_search_available_workers(worker_availability, current_time, next_worker_idx)
            
            base_threshold = max(0.01, task['waiting_time'] / (1 + np.log(1 + current_available_count)))
            with torch.no_grad():
                state = query_strategy.dqo_agent.get_state(
                    current_time,
                    task['location'],
                    current_available_count,
                    len(tasks)
                )
                action = query_strategy.dqo_agent.select_action(state)
            adjustment_ratio = (action - 3) * 0.1
            time_threshold = base_threshold * (1 + adjustment_ratio)
            time_threshold = np.clip(time_threshold, 0.05, 0.5)
        
        if task['waiting_time'] - current_time >= time_threshold:
            task_start_time = current_time
            available_workers = get_available_workers(local_workers, current_time)
            
            local_team = greedy_assignment(task, available_workers, current_time)
            task_revenue = calculate_real_task_revenue(task, local_team)
            local_revenue = task_revenue
            
            update_task = Update_task(task, local_team)
            for worker in local_team:
                update_worker(worker, task)

            local_skills = set()
            for worker in local_team:
                local_skills.update(worker['skills'])
            local_coverage = len(local_skills & set(task['skills'])) / len(task['skills'])

            if local_coverage < 1:  
                coop_available_workers = get_available_workers(coop_workers, current_time)
                coop_team = precision_query_range_optimization(
                    update_task, coop_available_workers, local_team, 
                    current_time, query_strategy, metrics, worker_velocity
                )
                
                if coop_team:
                    team = coop_team + local_team
                    new_revenue = calculate_real_task_revenue(task, team)
                    
                    comp_cost = len(coop_team) * 5
                    comm_cost = len(coop_team) * 2
                    total_cost = comp_cost + comm_cost
                    
                    revenue_increment = new_revenue - local_revenue
                    if revenue_increment > total_cost:
                        metrics.update_cooperation_cost(comp_cost, comm_cost)
                        coop_revenue = new_revenue
                        
                        for worker in coop_team:
                            update_worker(worker, task)
                    else:
                        coop_revenue = local_revenue
            else:
                coop_revenue = local_revenue
            
            task_completion_time = current_time + task['waiting_time']
            metrics.update_response_time(task_start_time, task_completion_time)
            metrics.update_revenue(coop_revenue)
            
            total_skills = len(task['skills'])
            covered_skills = total_skills - len(task['uncover_skills'])
            coverage_rate = covered_skills / total_skills
            metrics.update_skill_coverage(id(task), coverage_rate)
            
            task['coop_revenue'] = coop_revenue
            task['local_revenue'] = local_revenue
            finish_tasks.append(task)
            query_strategy.tasks_history.append(task)
            current_time = task['start_time']
        else:
            current_time += 0.01

    return finish_tasks, metrics 