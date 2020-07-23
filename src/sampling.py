import random
from collections import deque
from typing import List, Union
import numpy as np
from models import BetaBernoulli
from scipy.stats import beta

def random_sampling(deques: List[deque], 
                    weighted=False,
                    **kwargs) -> Union[int, List[int]]:
    while True:
        if weighted:
            weights = [len(deques[k]) for k in range(len(deques))]
            category = random.choices(np.arange(len(deques)), weights = weights)[0]
        else:
            category = random.randrange(len(deques))
        if len(deques[category]) != 0:
            return category
        
        
def thompson_sampling(deques: List[deque],
                      reward: List[float], #(num_groups, )
                      topk: int = 1, 
                     **kwargs) -> Union[int, List[int]]:
    ranked = np.argsort(reward)[::-1] # select the one with highest reward
    categories = []
    for category in ranked:
        if len(deques[category]) != 0:
            categories.append(category)
        if len(categories) == topk:
            break
    if topk == 1:
        return categories[0]
    else:
        return categories
        

SAMPLE_CATEGORY = {
    'random': random_sampling,
    'ts': thompson_sampling,
}
