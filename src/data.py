from data_utils import eval_ece, DATAFILE_DICT
import numpy as np
from typing import List, Tuple, Dict, Deque
from sklearn.metrics import confusion_matrix
from collections import deque
from sklearn.utils import shuffle 

NUM_BINS = 20

class Dataset:
    def __init__(self,
                 labels: np.ndarray,
                 scores: np.ndarray,
                 dataset_name: str) -> None:
        self.labels = labels
        self.scores = scores
        self.indices = np.arange(labels.shape[0])
        self._add_information(dataset_name)

    def __len__(self):
        return self.labels.shape[0]
    
    def _add_information(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_classes = self.scores.shape[-1]
        self.accuracy = (self.labels == self.predictions).mean()
        self.ece = eval_ece(np.max(self.scores,axis=-1), (self.labels == self.predictions), NUM_BINS)

    def enqueue(self) -> List[Deque[int]]:
        # group by self.categories
        queues = [deque() for _ in range(self.num_groups)]
        for index, label, score, category in zip(self.indices, self.labels, self.scores, self.categories):
            queues[category].append({'index': index, # might be used when logits need to be queries from a different file
                                     'label': label, 
                                     'score': score})
        return queues
    
    def group(self, group_method:str) -> None:
        if group_method == 'predicted_class':
            self.categories = self.predictions
            self.num_groups = self.num_classes
        elif group_method == 'score':
            self.num_groups = 10
            bins = np.linspace(0, 1, self.num_groups + 1)
            self.categories = np.digitize(np.max(self.scores,axis=-1), bins[1:-1])
        self._add_group_information()
    
    def _add_group_information(self) -> None:
        
        self.weight_k = np.array([(self.categories == idx).sum() * 1.0 / self.__len__()
                                 for idx in range(self.num_groups)])
        self.confidence_k = np.array([np.max(self.scores[self.categories == idx],axis=-1).mean()
                                 for idx in range(self.num_groups)])
        self.accuracy_k = np.array([((self.labels == self.predictions)[self.categories == idx]).mean()
                                 for idx in range(self.num_groups)])
        self.ece_k = np.array([eval_ece(np.max(self.scores[self.categories == idx],axis=-1), 
                                         (self.labels == self.predictions)[self.categories == idx], NUM_BINS)
                                    for idx in range(self.num_groups)])
        
    def shuffle(self, random_state=0) -> None:
        # To make sure the rows still align we shuffle an array of indices, and use these to
        # re-order the dataset's attributes.
        # Need to group before shuffle
        shuffle_ids = np.arange(self.labels.shape[0])
        shuffle_ids = shuffle(shuffle_ids, random_state=random_state)
        self.labels = self.labels[shuffle_ids]
        self.scores = self.scores[shuffle_ids]
        self.indices = self.indices[shuffle_ids]
        self.categories = self.categories[shuffle_ids]

    @classmethod
    def load_from_text(cls, dataset_name: str) -> 'Dataset':
        """
        Load dataset from a text file. Assumed format is:
            correct_class score_0 ... score_k
        """
        fname = DATAFILE_DICT[dataset_name]
        array = np.genfromtxt(fname)
        labels = array[:, 0].astype(np.int)
        scores = array[:, 1:].astype(np.float)
        return cls(labels, scores, dataset_name)

    @property
    def confusion_probs(self) -> np.ndarray:
        arr = confusion_matrix(self.labels, self.predictions).transpose()
        return arr / arr.sum(axis=-1, keepdims=True)

    @property
    def confusion_prior(self) -> np.ndarray:
        arr = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            arr[i] = self.scores[self.predictions == i].sum(axis=0) / sum(self.predictions == i)
        return arr

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.scores, axis=-1)
    
    

# def get_ground_truth(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
#                      metric: str, mode: str, topk: int = 1) -> np.ndarray:

#     if metric == 'accuracy':
#         metric_val = get_accuracy_k(categories, observations, num_classes)
#     elif metric == 'calibration_error':
#         metric_val = get_ece_k(categories, observations, confidences, num_classes, num_bins=10)
#     output = np.zeros((num_classes,), dtype=np.bool_)

#     if mode == 'max':
#         indices = metric_val.argsort()[-topk:]
#     else:
#         indices = metric_val.argsort()[:topk]
#     output[indices] = 1
#     return output


# def get_bayesian_ground_truth(categories: List[int], observations: List[bool], confidences: List[float],
#                               num_classes: int,
#                               metric: str, mode: str, topk: int = 1, pseudocount: int = 1, prior=None) -> np.ndarray:

#     if metric == 'accuracy':
#         model = BetaBernoulli(num_classes, prior=prior)
#         model.update_batch(confidences, observations)
#     elif metric == 'calibration_error':
#         model = ClasswiseEce(num_classes, num_bins=10, pseudocount=pseudocount)
#         model.update_batch(categories, observations, confidences)
#     metric_val = model.eval

#     output = np.zeros((num_classes,), dtype=np.bool_)
#     if mode == 'max':
#         indices = metric_val.argsort()[-topk:]
#     else:
#         indices = metric_val.argsort()[:topk]
#     output[indices] = 1

#     return output
