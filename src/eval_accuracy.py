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
from utils import mean_reciprocal_rank
import pickle

LOG_FREQ = 10
# method_list = ['random_arm', 'random_data', 'random_arm_informed', 'random_data_informed', 'ts_uniform', 'ts_informed']
method_list = ['random_data', 'random_data_informed', 'ts_informed']
DATASET_LIST = ['cifar100', 'dbpedia', '20newsgroup', 'svhn', 'imagenet'] #'imagenet', 


def eval_topk(mpe_log: np.ndarray, ground_truth: list, topk: int = 1) -> Dict[str, np.ndarray]:
    """
    Top k greatest
    :param mpe_log:(num_runs, num_samples // LOG_FREQ, num_classes)
    :param ground_truth: list of integers of length topk. Ground truth of topk classes.
    :param topk: int
    """
    assert len(ground_truth) == topk
    num_runs, num_evals, num_classes = mpe_log.shape
    avg_num_agreement = [None] * num_evals
    mrr = [None] * num_evals

    ground_truth_array = np.zeros((num_classes,), dtype=np.bool_)
    ground_truth_array[np.array(ground_truth).astype(int)] = 1

    for idx in range(num_evals):
        current_result = mpe_log[:, idx, :]
        topk_arms = np.argsort(current_result, axis=-1)[:, -topk:]
        topk_list = topk_arms.flatten().tolist()
        avg_num_agreement[idx] = len([arm for arm in topk_list if arm in ground_truth]) * 1.0 / (topk * num_runs)
        mrr[idx] = [mean_reciprocal_rank(mpe_log[run_id, idx, :], ground_truth_array, 'max') 
                        for run_id in range(num_runs)]
    return {
        'avg_num_agreement': np.array(avg_num_agreement),
        'mrr': np.array(mrr),
    }


def main():
    print('SAMPLING...')
    if args.topk:
        topk = TOPK_DICT[args.dataset_name]
    else:
        topk = 1
    output = pathlib.Path("../output/%s/" % args.metric )
    experiment_name = '%s_groupby_%s_top%d_pseudocount%.2f' % (args.dataset_name, args.group_method, topk, args.pseudocount)
    
    # load data
    dataset = Dataset.load_from_text(args.dataset_name)
    dataset.group(group_method = args.group_method)
    samples = {}
    mpe_log = {}
    # load results
    for method_name in method_list:
        mpe_log[method_name] = np.load(open(output / experiment_name / ('mpe_log_%s.npy' % method_name), 'rb'))

    
    print('EVAL GROUND TRUTH...')
    ground_truth = {}
    ground_truth['ece'] = dataset.ece
    ground_truth['accuracy_k'] = dataset.accuracy_k
    ground_truth['confidence_k'] = dataset.confidence_k
    ground_truth['weight_k'] = dataset.weight_k
    ground_truth['accuracy_k'][np.isnan(ground_truth['accuracy_k'])] = 0
    ground_truth['least_accurate'] = np.argsort(dataset.accuracy_k)[:topk]
    ground_truth['most_accurate'] = np.argsort(dataset.accuracy_k)[-topk:]
    
    print('EVAL L2 ERROR...')
    l2_error = {}
    l2_ece = {}
    l1_ece = {}
    ece = {}
    for method_name in method_list:
        error = mpe_log[method_name]-ground_truth['accuracy_k'][np.newaxis,np.newaxis,:]
        error[np.isnan(error)] = 0
        l2_error[method_name] = np.sqrt(np.inner((np.abs(error))**2, ground_truth['weight_k']))

        error = mpe_log[method_name]-ground_truth['confidence_k'][np.newaxis,np.newaxis,:]
        error[np.isnan(error)] = 0
        l2_ece[method_name] = np.inner((np.abs(error)**2), ground_truth['weight_k'])
        l1_ece[method_name] = np.inner((np.abs(error)), ground_truth['weight_k'])
        
        offset = mpe_log[method_name]-ground_truth['confidence_k'][np.newaxis,np.newaxis,:]
        ece[method_name] = np.inner((np.abs(offset)), ground_truth['weight_k'])
        
    pickle.dump(ground_truth, open(output / experiment_name / "ground_truth.pkl", "wb"))  
    pickle.dump(l2_error, open(output / experiment_name / "l2_error.pkl", "wb")) 
    pickle.dump(l2_ece, open(output / experiment_name / "l2_ece.pkl", "wb"))
    pickle.dump(l1_ece, open(output / experiment_name / "l1_ece.pkl", "wb"))
    pickle.dump(ece, open(output / experiment_name / "ece.pkl", "wb"))
    
        
    if args.metric in ['most_accurate', 'least_accurate']:
        print('EVAL RANKING...')
        avg_num_agreement = {}
        mrr = {}
        for method_name in method_list:
            if args.metric  == 'least_accurate':
                eval_topk_output = eval_topk(-mpe_log[method_name],
                                            ground_truth[args.metric], topk=topk)
            elif args.metric == 'most_accurate':
                eval_topk_output = eval_topk(mpe_log[method_name],
                                            ground_truth[args.metric], topk=topk)
            avg_num_agreement[method_name] = eval_topk_output['avg_num_agreement'] # num_samples // LOG_FREQ
            mrr[method_name] = eval_topk_output['mrr'] # num_runs, num_samples // LOG_FREQ
     
        pickle.dump(avg_num_agreement, open(output / experiment_name / "avg_num_agreement.pkl", "wb")) 
        pickle.dump(mrr, open(output / experiment_name / "mrr.pkl", "wb")) 

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='cifar100')
    parser.add_argument('metric', type=str, default='groupwise_accuracy')
    parser.add_argument('group_method', type=str, default='predicted_class')
    parser.add_argument('pseudocount', type=float, default=2)
    parser.add_argument('topk', type=str, default='True')
    
    args, _ = parser.parse_known_args()
    args.topk = args.topk  == 'True'
    
    main()
