import argparse
import pathlib
import random
from collections import deque
from typing import List, Dict, Tuple, Union
from data import Dataset
from data_utils import *
from sampling import *
from models import BetaBernoulli
import numpy as np
from tqdm import tqdm
LOG_FREQ = 10
output_dir = pathlib.Path("../output")


def select_and_label(dataset: 'Dataset', sample_method: str, prior=None, weighted=False, topk=1) -> np.ndarray:

    model = BetaBernoulli(dataset.num_groups, prior=prior, weight=dataset.weight_k)
    deques = dataset.enqueue()
    dataset_len = dataset.__len__()
    dataset_num_groups = dataset.num_groups
    del dataset

    sampled_indices = np.zeros((dataset_len,), dtype=np.int) # indices of selected data points
    mpe_log = np.zeros((dataset_len // LOG_FREQ, dataset_num_groups))
    
    sample_fct = SAMPLE_CATEGORY[sample_method]

    idx = 0
    mpe_log[0] = model.mpe
    while idx < dataset_len:
        if sample_method == 'ts':
            reward = model.reward(reward_type=args.metric)
        else:
            # no need to compute reward for non-ts methods
            reward=None
        categories = sample_fct(deques=deques, reward=reward, weighted=weighted, topk=topk)
        if topk == 1 or sample_method != 'ts':
            categories = [categories]
        for category in categories:
            selected = deques[category].pop() # a dictionary
            model.update(category, selected)
            sampled_indices[idx] = selected['index']
            if (idx+1) % LOG_FREQ == 0:
                mpe_log[idx // LOG_FREQ] = model.mpe
            idx += 1

    return sampled_indices,  mpe_log

def main():
    
    if args.dataset_name == 'imagenet':
        RUNS = 50
    else:
        RUNS = 1000
                                  
    experiment_name = '%s_groupby_%s_top%d_pseudocount%.2f' % (args.dataset_name, args.group_method, args.topk, args.pseudocount)
    if not (output_dir /args.metric).is_dir():
        (output_dir /args.metric).mkdir()
    if not (output_dir /args.metric/ experiment_name).is_dir():
        (output_dir /args.metric/ experiment_name).mkdir()
    method_list = ['random_arm', 'random_data', 'random_arm_informed', 'random_data_informed', 'ts_uniform', 'ts_informed']
    
    samples = {}
    mpe_log = {}
    dataset = Dataset.load_from_text(args.dataset_name)
    dataset.group(group_method = args.group_method)
    
    UNIFORM_PRIOR = np.ones((dataset.num_groups, 2)) / 2
    INFORMED_PRIOR = np.array([dataset.confidence_k, 1 - dataset.confidence_k]).T
    INFORMED_PRIOR[np.isnan(INFORMED_PRIOR)] = 1.0 / 2 
    config_dict = {
        'random_arm': [UNIFORM_PRIOR * 1e-6, 'random', False],
        'random_data': [UNIFORM_PRIOR * 1e-6, 'random', True],
        'random_arm_informed': [INFORMED_PRIOR * args.pseudocount, 'random', False],
        'random_data_informed': [INFORMED_PRIOR * args.pseudocount, 'random', True],
        'ts_uniform': [UNIFORM_PRIOR * args.pseudocount, 'ts', None], 
        'ts_informed': [INFORMED_PRIOR * args.pseudocount, 'ts', None]}
        
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
                                                                   prior=prior, weighted=weighted, topk=args.topk)
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
    parser.add_argument('pseudocount', type=float, default=2)
    parser.add_argument('topk', type=int, default=1)
    
    args, _ = parser.parse_known_args()
    main()