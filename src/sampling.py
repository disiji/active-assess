import random
from collections import deque
from typing import List, Union

import numpy as np

from models import BetaBernoulli


def random_sampling(deques: List[deque], topk: int = 1, **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with random sampling.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
    while True:
        # select each class randomly
        if topk == 1:
            category = random.randrange(len(deques))
            if len(deques[category]) != 0:
                return category
        else:
            # return a list of randomly selected categories:
            candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
            if len(candidates) < topk:
                return random_sampling(deques, topk=1)
            else:  # there are less than topk available arms to play
                return random.sample(candidates, topk)


def thompson_sampling(deques: List[deque],
                      model,
                      mode: str,
                      topk: int = 1,
                      **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with Thompson sampling.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param model: BetaBernoulli, SumOfBetaEce
        A model for classwise accuracy.
    :param mode: str
        'min' or 'max'
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
    # 
    samples = model.sample()
    if mode == 'max':
        ranked = np.argsort(samples)[::-1]
    elif mode == 'min':
        ranked = np.argsort(samples)
    if topk == 1:
        for category in ranked:
            if len(deques[category]) != 0:
                return category
    else:
        categories_list = []

        candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
        # when we go through 'ranked' and len(categories_list) < topk, topk sampling is reduced to top 1
        if len(candidates) < topk:
            return thompson_sampling(deques, model, mode, topk=1)
        else:
            for category in ranked:
                if category in candidates:
                    categories_list.append(category)
                    if len(categories_list) == topk:
                        return categories_list


SAMPLE_CATEGORY = {
    'random': random_sampling,
    'ts': thompson_sampling,
}
