# COTF Cross-Platform Online Team Formation in Spatial Crowdsourcing

COTF is a novel framework for dynamic task allocation in spatial crowdsourcing, focusing on efficient worker cooperation and skill coverage optimization.

## Overview

The COTF framework addresses the challenges of real-time task allocation in spatial crowdsourcing by:
- Managing dynamic worker-task matching
- Optimizing skill coverage through worker cooperation
- Balancing revenue and cooperation costs
- Implementing intelligent query strategies (PQROS and DQO)

## Key Features

- **Dynamic Task Allocation**: Real-time assignment of tasks to workers based on:
  - Spatial constraints
  - Skill requirements
  - Time windows
  - Worker availability

- **Worker Cooperation Strategies**:
  - Local worker teams
  - Cooperative worker teams
  - Skill coverage optimization
  - Cost-effective cooperation decisions

- **Query Strategies**:
  - PQROS (Precision Query Range Optimization Strategy)
  - DQO (Deep Q-learning Optimization)


## Parameters

- `num_tasks`: Number of tasks to generate
- `num_local_workers`: Number of local workers
- `num_coop_workers`: Number of cooperative workers
- `worker_velocity`: Movement speed of workers
- `strategy_type`: Query strategy type ("PQROS" or "DQO")

## Usage

```python

cd COTF
python main.py

## Future Improvements

1. Dynamic worker velocity adjustment
2. Adaptive cooperation cost parameters
3. Enhanced skill matching algorithms
4. Multi-objective optimization
5. Real-time worker reallocation

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- Random
- Collections 
