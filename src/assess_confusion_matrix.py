import argparse
import pathlib
import random
from collections import deque
from typing import List, Dict, Tuple, Union
from data import Dataset, SuperclassDataset
from data_utils import *
from sampling import *
from models import BetaBernoulli
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import matplotlib;matplotlib.rcParams['font.size'] = 10
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'

LOG_FREQ = 10
output_dir = pathlib.Path("../output/confusion_matrix")
group_method = 'predicted_class'

def select_and_label(dataset: 'Dataset', sample_method: str, budget: int, costs:np.ndarray, \
                     prior=None, weighted=False) -> np.ndarray:
    
    model = DirichletMultinomial(prior, costs, weight=dataset.weight_k)
    deques = dataset.enqueue()
    sample_fct = SAMPLE_CATEGORY[sample_method]
    
    sampled_indices = np.zeros((budget,), dtype=np.int) # indices of selected data points
    mpe_log = np.zeros((budget // LOG_FREQ, dataset.num_groups, dataset.num_groups))

    for idx in tqdm(range(budget)):
        if sample_method == 'ts':
            reward = model.reward(reward_type='confusion_matrix')
            category = sample_fct(deques=deques, reward=reward)
        else:
            category = sample_fct(deques=deques, weighted=weighted)
        selected = deques[category].pop() # a dictionary
        model.update(category, selected)
        sampled_indices[idx] = selected['index']
        if (idx+1) % LOG_FREQ == 0:
            mpe_log[idx // LOG_FREQ] = model.mpe 
            
    return {'sampled_indices': sampled_indices,  
            'mpe_log': mpe_log}

def main():
    if args.superclass:
        experiment_name = '%s_superclass_pseudocount%d' % (args.dataset_name, args.pseudocount)
        dataset = SuperclassDataset.load_from_text(args.dataset_name, CIFAR100_SUPERCLASS_LOOKUP)
    else:
        experiment_name = '%s_pseudocount%d' % (args.dataset_name, args.pseudocount)
        dataset = Dataset.load_from_text(args.dataset_name)
    if not (output_dir / experiment_name).is_dir():
        (output_dir / experiment_name).mkdir()
    method_list = ['random_arm', 'random_data', 'ts_uniform', 'ts_informed']
    
    dataset.group(group_method = group_method)
    deques = dataset.enqueue()
    budget = dataset.__len__()
    costs = np.ones((dataset.num_classes, dataset.num_classes))
    np.fill_diagonal(costs, 0)
    
    UNIFORM_PRIOR = np.ones((dataset.num_groups, dataset.num_groups)) / dataset.num_groups * args.pseudocount
    INFORMED_PRIOR = args.pseudocount * dataset.confusion_prior
    
    config_dict = {
        'random_arm': [UNIFORM_PRIOR * 1e-6, 'random', False],
        'random_data': [UNIFORM_PRIOR * 1e-6, 'random', True],
        'ts_uniform': [UNIFORM_PRIOR, 'ts', None], 
        'ts_informed': [INFORMED_PRIOR, 'ts', None]}

    for r in tqdm(range(args.run_start, args.run_end)):
        dataset = Dataset.load_from_text(args.dataset_name)
        dataset.group(group_method = group_method)
        dataset.shuffle(r)
        for method_name in method_list:
            prior, sample_method, weighted = config_dict[method_name]
            output = select_and_label(dataset, sample_method=sample_method, budget=budget, costs=costs, \
                                      prior=prior, weighted=weighted)  
            samples = output['sampled_indices'] #(budget, )
            mpe_log = output['mpe_log'] #(dataset.num_groups, dataset.num_groups)
            # write samples to file
            np.save(open(output_dir / experiment_name / ('samples_%s_run%d.npy' % (method_name, r)), 'wb'), samples)
            np.save(open(output_dir / experiment_name / ('mpe_log_%s_run%d.npy' % (method_name, r)), 'wb'), mpe_log)
    return samples, mpe_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='cifar100')
    parser.add_argument('superclass', type=str, default='False')
    parser.add_argument('run_start', type=int, default=0)
    parser.add_argument('run_end', type=int, default=100)
    parser.add_argument('pseudocount', type=int, default=1)
    
    args, _ = parser.parse_known_args()
    args.superclass = args.superclass == 'True'
    print(args.superclass)
    main()