import argparse
import pathlib
import random
from collections import deque
from typing import List, Dict, Tuple, Union
from data import Dataset, SuperclassDataset
from data_utils import *
from models import BetaBernoulli
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sampling import *

LOG_FREQ = 10
output_dir = pathlib.Path("../output/difference_random_2_groups")

RUNS = 1000
random.seed(1234)
rope_width = 0.05

def rope(alpha0, alpha1, beta0, beta1):
    num_samples = 10000
    theta_0 = np.random.beta(alpha0, beta0, size=(num_samples))
    theta_1 = np.random.beta(alpha1, beta1, size=(num_samples))
    delta = theta_0 - theta_1
    return [(delta < -rope_width).mean(), (np.abs(delta) <= rope_width).mean(), (delta > rope_width).mean()]

def select_and_label(dataset: 'Dataset', sample_method: str, budget: int, group0: int, group1: int, \
                     prior=None, weighted=False) -> np.ndarray:
    
    model = BetaBernoulli(dataset.num_groups, prior=prior, weight=dataset.weight_k)
    deques = dataset.enqueue()

    for i in range(len(deques)):
        if i not in [group0, group1]:
            deques[i].clear()
            
    sampled_indices = np.zeros((budget,), dtype=np.int) # indices of selected data points
    mpe_log = np.zeros((budget // LOG_FREQ, dataset.num_groups))
    rope_eval = np.zeros((budget // LOG_FREQ, 3))
    
    sample_fct = SAMPLE_CATEGORY[sample_method]

    idx = 0
    while idx < budget:
        if sample_method == 'ts':
            reward = model.reward(reward_type='difference', group0 = group0, group1 = group1)
            category = sample_fct(deques=deques, reward=reward)
        else:
            category = sample_fct(deques=deques, weighted=weighted)
        selected = deques[category].pop() # a dictionary
        model.update(category, selected)
        sampled_indices[idx] = selected['index']
        if (idx+1) % LOG_FREQ == 0:
            mpe_log[idx // LOG_FREQ] = model.mpe
            alpha0, beta0 = model._params[group0]
            alpha1, beta1 = model._params[group1]
            rope_eval[idx // LOG_FREQ] = rope(alpha0, alpha1, beta0, beta1)           
        idx += 1
#         if idx == budget:
#             print(alpha0, beta0, alpha1, beta1)
    return {'sampled_indices': sampled_indices,  
            'mpe_log': mpe_log,
            'rope_eval': rope_eval}

def main():
    if args.group0 != -1 and args.group1!= -1:
        experiment_name = '%s_groupby_%s_group0_%d_group1_%d_pseudocount%.2f' % (args.dataset_name, args.group_method,\
                                                                                 args.group0, args.group1, args.pseudocount)
    else:
        experiment_name = '%s_groupby_%s_pseudocount%.2f' % (args.dataset_name, args.group_method, args.pseudocount)
    if not (output_dir / experiment_name).is_dir():
        (output_dir / experiment_name).mkdir()
    method_list = ['random_arm_symmetric', 'random_data_symmetric', \
                   'random_arm_informed', 'random_data_informed', \
                   'ts_uniform', 'ts_informed']
    
    samples = {}
    mpe_log = {}
    rope_eval = {}
  
    
    if args.dataset_name == 'superclass_cifar100':
        superclass = True
        dataset_name = 'cifar100'
        dataset = SuperclassDataset.load_from_text('cifar100', CIFAR100_SUPERCLASS_LOOKUP)
    else:
        superclass = False
        dataset = Dataset.load_from_text(args.dataset_name)
        
    dataset.group(group_method = args.group_method)    
    deques = dataset.enqueue()
    
    UNIFORM_PRIOR = np.ones((dataset.num_groups, 2)) / 2
    INFORMED_PRIOR = np.array([dataset.confidence_k, 1 - dataset.confidence_k]).T
    INFORMED_PRIOR[np.isnan(INFORMED_PRIOR)] = 1.0 / 2 
    config_dict = {
        'random_arm_symmetric': [UNIFORM_PRIOR * args.pseudocount, 'random', False],
        'random_data_symmetric': [UNIFORM_PRIOR * args.pseudocount, 'random', True],
        'random_arm_informed': [INFORMED_PRIOR * args.pseudocount , 'random', False],
        'random_data_informed': [INFORMED_PRIOR * args.pseudocount, 'random', True],
        'ts_uniform': [UNIFORM_PRIOR * args.pseudocount, 'ts', None], 
        'ts_informed': [INFORMED_PRIOR * args.pseudocount, 'ts', None]}

    max_budget = np.sort(np.array([len(i) for i in deques]))[-2:].sum()
    print(max_budget)
    for method in method_list:
        samples[method] = np.zeros((RUNS, max_budget), dtype=np.int) 
        mpe_log[method] = np.zeros((RUNS, max_budget // LOG_FREQ, dataset.num_groups))
        rope_eval[method] = np.zeros((RUNS, max_budget // LOG_FREQ, 3))
    configs = np.zeros((RUNS, 4))
    
    for r in tqdm(range(RUNS)):
        
        if args.dataset_name == 'superclass_cifar100':
            superclass = True
            dataset_name = 'cifar100'
            dataset = SuperclassDataset.load_from_text('cifar100', CIFAR100_SUPERCLASS_LOOKUP)
        else:
            superclass = False
            dataset = Dataset.load_from_text(args.dataset_name)
        
        dataset.group(group_method = args.group_method)
        dataset.shuffle(r)

        tmp = np.arange(dataset.num_groups)
        random.Random(r).shuffle(tmp)
        if args.group0 == -1:
            group0 = tmp[0]
        else:
            group0 = args.group0
        if args.group1 == -1:
            group1 = tmp[1]
        else:
            group1 = args.group1
            
        budget = len(deques[group0]) + len(deques[group1])  
        delta = dataset.accuracy_k[group0] - dataset.accuracy_k[group1]
        
        configs[r] = np.array([group0, group1, budget, delta])
        
        for method_name in method_list:
            
            prior, sample_method, weighted = config_dict[method_name]
            output = select_and_label(dataset, sample_method=sample_method, budget=budget, group0=group0, group1=group1,
                                      prior=prior, weighted=weighted)  
            samples[method_name][r][:budget] = output['sampled_indices']
            mpe_log[method_name][r][:budget// LOG_FREQ]  = output['mpe_log']
            rope_eval[method_name][r][:budget// LOG_FREQ]  = output['rope_eval']
            
    # write samples to file
    for method in method_list:
        np.save(open(output_dir / experiment_name / ('samples_%s.npy' % method), 'wb'), samples[method])
        np.save(open(output_dir / experiment_name / ('mpe_log_%s.npy' % method), 'wb'), mpe_log[method])
        np.save(open(output_dir / experiment_name / ('rope_eval_%s.npy' % method), 'wb'), rope_eval[method])
    np.save(open(output_dir / experiment_name / 'configs.npy', 'wb'), configs)
    return samples, mpe_log, rope_eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='cifar100')
    parser.add_argument('group_method', type=str, default='predicted_class')
    parser.add_argument('pseudocount', type=float, default=2)
    parser.add_argument('group0', type=int, default=-1)
    parser.add_argument('group1', type=int, default=-1)
    
    args, _ = parser.parse_known_args()
    main()