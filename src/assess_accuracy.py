import argparse
import pathlib
import random
from collections import deque
from typing import List, Dict, Tuple, Union
from data import Dataset
from data_utils import *
from sampling import *
from models import BetaBernoulli
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import matplotlib;matplotlib.rcParams['font.size'] = 10
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'

LOG_FREQ = 10
output_dir = pathlib.Path("../output")
RUNS = 100

def select_and_label(dataset: 'Dataset', sample_method: str, prior=None, weighted=False) -> np.ndarray:

    model = BetaBernoulli(dataset.num_groups, prior=prior, weight=dataset.weight_k)
    deques = dataset.enqueue()

    sampled_indices = np.zeros((dataset.__len__(),), dtype=np.int) # indices of selected data points
    mpe_log = np.zeros((dataset.__len__() // LOG_FREQ, dataset.num_groups))
    
    sample_fct = SAMPLE_CATEGORY[sample_method]

    idx = 0
    while idx < dataset.__len__():
        if sample_method == 'ts':
            reward = model.reward(reward_type=args.metric)
        else:
            reward=None
        category = sample_fct(deques=deques, reward=reward, weighted=weighted)
        selected = deques[category].pop() # a dictionary
        model.update(category, selected)
        sampled_indices[idx] = selected['index']
        if (idx+1) % LOG_FREQ == 0:
            mpe_log[idx // LOG_FREQ] = model.mpe
        idx += 1

    return sampled_indices,  mpe_log

def main():
                                  
    experiment_name = '%s_groupby_%s_pseudocount%.2f' % (args.dataset_name, args.group_method, args.pseudocount)
    if not (output_dir /args.metric).is_dir():
        (output_dir /args.metric).mkdir()
    if not (output_dir /args.metric/ experiment_name).is_dir():
        (output_dir /args.metric/ experiment_name).mkdir()
    method_list = ['random_arm', 'random_data', 'ts_uniform', 'ts_informed']
    
    samples = {}
    mpe_log = {}
    dataset = Dataset.load_from_text(args.dataset_name)
    dataset.group(group_method = args.group_method)
    
    UNIFORM_PRIOR = np.ones((dataset.num_groups, 2)) / 2 * args.pseudocount
    INFORMED_PRIOR = np.array([dataset.confidence_k, 1 - dataset.confidence_k]).T * args.pseudocount
    INFORMED_PRIOR[np.isnan(INFORMED_PRIOR)] = 1.0 / 2 * args.pseudocount
    config_dict = {
        'random_arm': [UNIFORM_PRIOR * 1e-6, 'random', False],
        'random_data': [UNIFORM_PRIOR * 1e-6, 'random', True],
        'ts_uniform': [UNIFORM_PRIOR, 'ts', None], 
        'ts_informed': [INFORMED_PRIOR, 'ts', None]}
        
    for method in method_list:
        samples[method] = np.zeros((RUNS, dataset.__len__()), dtype=np.int) 
        mpe_log[method] = np.zeros((RUNS, dataset.__len__() // LOG_FREQ, dataset.num_groups))

    for r in tqdm(range(RUNS)):
        
        dataset = Dataset.load_from_text(args.dataset_name)
        dataset.group(group_method = args.group_method)
        dataset.shuffle(r)

        for method_name in method_list:
            prior, sample_method, weighted = config_dict[method_name]
            samples[method_name][r], mpe_log[method_name][r] = select_and_label(dataset, sample_method=sample_method, 
                                                                   prior=prior, weighted=weighted)
    # write samples to file
    for method in method_list:
        np.save(open(output_dir /args.metric/ experiment_name/ ('samples_%s.npy' % method), 'wb'), samples[method])
        np.save(open(output_dir /args.metric/ experiment_name / ('mpe_log_%s.npy' % method), 'wb'), mpe_log[method])
    return samples, mpe_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='cifar100')
    parser.add_argument('group_method', type=str, default='predicted_class')
    parser.add_argument('metric', type=str, default='groupwise_accuracy')
    parser.add_argument('pseudocount', type=int, default=2)
    
    args, _ = parser.parse_known_args()
    main()