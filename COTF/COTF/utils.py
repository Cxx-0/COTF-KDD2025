import numpy as np
from collections import Counter

def calculate_distance(loc1, loc2):
    return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

def can_reach(worker, task, current_time):
    distance = calculate_distance(worker['location'], task['location'])
    travel_time = distance / worker['velocity']
    return (task['waiting_time'] - (current_time - worker['available_time'])) >= travel_time

def update_worker(worker, task):
    worker['location'] = task['location']
    worker['available_time'] = task['start_time'] + task['waiting_time']
    worker['assigned'] = True

def calculate_real_task_revenue(task, team):
    if not team:
        return 0
    
    covered_skills = set()
    for worker in team:
        covered_skills.update(worker['skills'])
    
    coverage_rate = len(covered_skills & set(task['skills'])) / len(task['skills'])
    base_revenue = task['base_revenue']
    
    return base_revenue * coverage_rate

def Update_task(task, team):
    updated_task = task.copy()
    covered_skills = set()
    for worker in team:
        covered_skills.update(worker['skills'])
    updated_task['uncover_skills'] = set(task['skills']) - covered_skills
    return updated_task

def calculate_max_query_range(task, current_time, worker_velocity):
    remaining_time = task['waiting_time'] - current_time
    if remaining_time <= 0:
        return 0
    return worker_velocity * remaining_time

def calculate_intersection_area(range1, range2, center1, center2):
    d = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    if d >= range1 + range2:
        return 0
    
    if d <= abs(range1 - range2):
        return np.pi * min(range1, range2)**2
    
    a = np.arccos((range1**2 + d**2 - range2**2) / (2 * range1 * d))
    b = np.arccos((range2**2 + d**2 - range1**2) / (2 * range2 * d))
    
    area = (range1**2 * a + range2**2 * b - 
            d * range1 * np.sin(a))
    
    return area

def calculate_expected_revenue(intersection_area_ratio, task, team):
    if not team:
        return 0
    
    covered_skills = set()
    for worker in team:
        covered_skills.update(worker['skills'])
    
    coverage_rate = len(covered_skills & set(task['skills'])) / len(task['skills'])
    base_revenue = task['base_revenue']
    
    return base_revenue * coverage_rate * intersection_area_ratio

def get_available_workers(workers, current_time):
    available_workers = []
    for worker in workers:
        if worker['available_time'] <= current_time and not worker['assigned']:
            available_workers.append(worker)
    return available_workers 