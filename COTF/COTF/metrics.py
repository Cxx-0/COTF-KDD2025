import numpy as np
from collections import defaultdict

class Metrics:
    def __init__(self):
        self.ptr = 0
        self.tcc = 0
        self.response_times = []
        self.query_counts = 0
        self.comp_costs = 0
        self.comm_costs = 0
        self.skill_coverage = {}

    def update_revenue(self, revenue):
        self.ptr += revenue

    def update_cooperation_cost(self, comp_cost, comm_cost):
        self.tcc += comp_cost + comm_cost
        self.comp_costs += comp_cost
        self.comm_costs += comm_cost

    def update_response_time(self, start_time, completion_time):
        response_time = completion_time - start_time
        self.response_times.append(response_time)

    def update_query_count(self):
        self.query_counts += 1

    def update_skill_coverage(self, task_id, coverage_rate):
        self.skill_coverage[task_id] = coverage_rate

    def get_average_response_time(self):
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)

    def get_average_skill_coverage(self):
        if not self.skill_coverage:
            return 0
        return sum(self.skill_coverage.values()) / len(self.skill_coverage)

    def get_summary(self):
        return {
            'platform_total_revenue': self.ptr,
            'total_cooperation_cost': self.tcc,
            'average_response_time': self.get_average_response_time(),
            'query_counts': self.query_counts,
            'computation_costs': self.comp_costs,
            'communication_costs': self.comm_costs,
            'average_skill_coverage': self.get_average_skill_coverage()
        } 