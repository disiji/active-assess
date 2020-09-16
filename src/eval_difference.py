import argparse
import pathlib
import random
from collections import deque
from typing import List, Dict, Tuple, Union
from data import Dataset, SuperclassDataset
from data_utils import *
from models import BetaBernoulli
import numpy as np
from tqdm import tqdm
from sampling import *
from scipy.stats import wilcoxon

LINEWIDTH = 13.97
LOG_FREQ = 1
output_dir = pathlib.Path("../output/difference_random_2_groups")


DATASET_LIST = ['superclass_cifar100', 'svhn', '20newsgroup', 'dbpedia'] #
method_list = ['random_data_symmetric', 'random_data_informed', 'ts_informed']
method_format = {#'random_arm_symmetric': ('Random Arm Symmetric', 'g', '.', '--'), 
                 'random_data_symmetric': ('UPrior', 'b', '^', '--'), 
                 #'random_arm_informed': ('Random Arm Informed', 'g', '.', '-'), 
                 'random_data_informed': ('IPrior', 'b', '^', '-'), 
                 #'ts_uniform': ('TS Symmetric', 'k', '*', '-'), 
                 'ts_informed': ('IPrior+TS', 'r', '+', '-'),
                }

metric = 'difference'
group_method = 'predicted_class'
pseudocount = 2
rope_width = 0.05


configs = {}
samples = {}
mpe_log = {}
rope_eval = {}

for dataset_name in tqdm(DATASET_LIST): # takes 4 minutes to load results of imagenet
    experiment_name = '%s_groupby_%s_pseudocount%.2f' % (dataset_name, group_method, pseudocount)
    samples[dataset_name], mpe_log[dataset_name], rope_eval[dataset_name] = {}, {}, {}
    
    configs[dataset_name] = np.load(open(output_dir / experiment_name / 'configs.npy', 'rb'))
    for method in method_list:
        rope_eval[dataset_name][method] = np.load(open(output_dir / experiment_name / \
                                                       ('rope_eval_%s.npy' % method), 'rb'))
        
        
def rope(alpha0, alpha1, beta0, beta1):
    num_samples = 1000
    theta_0 = np.random.beta(alpha0, beta0, size=(num_samples))
    theta_1 = np.random.beta(alpha1, beta1, size=(num_samples))
    delta = theta_0 - theta_1
    return [(delta < -rope_width).mean(), (np.abs(delta) <= rope_width).mean(), (delta > rope_width).mean()]

counts = dict()
budgets = dict()
rope_ground_truth_dict = dict()

for i, dataset_name in enumerate(DATASET_LIST):
    
    runs = configs[dataset_name].shape[0]
        
    counts[dataset_name] = {}
    budgets[dataset_name] = np.zeros((runs,))
    rope_ground_truth_dict[dataset_name] = np.zeros((runs,))
    for method_name in method_format:  
        counts[dataset_name][method_name] = []

    if dataset_name == 'superclass_cifar100':
        superclass = True
        dataset = SuperclassDataset.load_from_text('cifar100', CIFAR100_SUPERCLASS_LOOKUP)
    else:
        superclass = False
        dataset = Dataset.load_from_text(dataset_name)
        
    dataset.group(group_method = group_method)        
    dataset_len = dataset.__len__()
    dataset_accuracy_k = dataset.accuracy_k
    dataset_weight_k = dataset.weight_k
    del dataset
        
    for r in tqdm(range(runs)):
        group0, group1, budget, delta = configs[dataset_name][r]
        group0, group1, budget = int(group0), int(group1), int(budget)
        budgets[dataset_name][r] = budget
        rope_ground_truth = rope(dataset_len * dataset_weight_k[group0] * (dataset_accuracy_k[group0]+ 1e-6), 
             dataset_len * dataset_weight_k[group1] * (dataset_accuracy_k[group1]+ 1e-6), 
             dataset_len * dataset_weight_k[group0] * (1-dataset_accuracy_k[group0] + 1e-6),
             dataset_len * dataset_weight_k[group1] * (1-dataset_accuracy_k[group1] + 1e-6))

        if delta < -rope_width:
            rope_region = 0
        elif delta > rope_width:
            rope_region = 2
        else:
            rope_region = 1
            
        rope_ground_truth_dict[dataset_name][r] = rope_ground_truth[rope_region]
        
        for method_name in method_format:  
            rope_ = rope_eval[dataset_name][method_name][r,:budget//LOG_FREQ, rope_region]
            error_rate = np.abs(rope_ - rope_ground_truth[rope_region]) / rope_ground_truth[rope_region]
            error = (error_rate < 0.05)
            counts[dataset_name][method_name].append(((np.argmax(error[0:])+0)*LOG_FREQ+LOG_FREQ))
            #counts[dataset_name][method_name].append(np.abs(rope_[15] - rope_ground_truth[rope_region]))
            
val = np.zeros((len(DATASET_LIST), len(method_format)))
for i, dataset_name in enumerate(DATASET_LIST):
    tmp = []
    for method_name in method_format:
        tmp.append(np.mean(counts[dataset_name][method_name]))
    val[i] = np.array(tmp)
df = pd.DataFrame(val.T, 
                  index=[method_format[i][0] for i in method_format], 
                  columns=[DATASET_NAMES[dataset_name] for dataset_name in DATASET_LIST])

print('\\begin{tabular}{@{}ccccc@{}}')
print('\\toprule ')
for method in method_list:
    print('& {%10s}' % method_format[method][0], end = '')
print('\\\ \midrule')

for i, dataset_name in enumerate(DATASET_LIST):
    vals = []
    for method in method_list:
        vals.append(df[DATASET_NAMES[dataset_name]][method_format[method][0]])
    print('{%20s} & %.1f &%.1f &\\textbf{%.1f} \\\\ \n' % (DATASET_NAMES[dataset_name], vals[0], vals[1], vals[2]), end = '')
print('\\bottomrule')
print('\\end{tabular}')