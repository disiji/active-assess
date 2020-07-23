from data_utils import eval_ece, DATAFILE_DICT
import numpy as np
from typing import List, Tuple, Dict, Deque, Iterable
from sklearn.metrics import confusion_matrix
from collections import deque, defaultdict
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
        elif group_method == 'score_equal_width':
            self.num_groups = 10
            bins = np.linspace(0, 1, self.num_groups + 1)
            self.categories = np.digitize(np.max(self.scores,axis=-1), bins[1:-1]).astype(int)
        elif group_method == 'score_equal_size':
            self.num_groups = 10
            bin_size = self.__len__() // self.num_groups
            self.categories = np.zeros((self.__len__(),)).astype(int)
            ranked = np.argsort(np.max(self.scores,axis=-1))
            for idx in range(self.num_groups):
                start = idx * bin_size
                if idx == self.num_groups - 1:
                    end = -1
                else:
                    end = (idx+1) * bin_size
                self.categories[ranked[start : end]] = idx
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
        
    def get_labeled_datapoint(self, sample_index):
        pos = self.indices.tolist().index(sample_index)
        return {'index': sample_index, # might be used when logits need to be queries from a different file
                 'label': self.labels[pos], 
                 'score': self.scores[pos],
                'category': self.categories[pos]}

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
    def confusion_probs(self) -> np.ndarray:# use labels, grouped by predicted class
        arr = confusion_matrix(self.labels, self.predictions).transpose()
        return arr / arr.sum(axis=-1, keepdims=True)

    @property
    def confusion_prior(self) -> np.ndarray:
        arr = np.zeros((self.num_groups, self.num_groups))
        for i in range(self.num_groups):
            arr[i] = self.scores[self.predictions == i].sum(axis=0) / sum(self.predictions == i)
        return arr

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.scores, axis=-1)

    
class SuperclassDataset(Dataset):
    def __init__(self,
                 labels: np.ndarray,
                 scores: np.ndarray,
                 dataset_name: str,
                 superclass_lookup: Dict[int, int]) -> None:
        self.labels = labels
        self.scores = scores
        self.indices = np.arange(labels.shape[0])
        self._add_information(dataset_name)
        
        self.superclass_lookup = superclass_lookup
        self.reverse_lookup = defaultdict(list)
        for key, value in self.superclass_lookup.items():
            self.reverse_lookup[value].append(key)
        # group with superclass
        self.categories = np.array([self.superclass_lookup[class_idx] for class_idx in self.predictions])
        self.num_groups = len(self.reverse_lookup)
        # compute superclass scores
        self.superclass_scores = np.zeros((labels.shape[0], self.num_groups))
        for superclass_idx in range(self.num_groups):
            self.superclass_scores[:, superclass_idx] = np.sum(self.scores[:, self.reverse_lookup[superclass_idx]], axis=1)
        # compute superclass scores
        self._add_group_information()

    def enqueue(self) -> List[Deque[int]]:
        # group by self.categories
        queues = [deque() for _ in range(self.num_groups)]
        for index, label, superclass_score, category in zip(self.indices, self.labels, self.superclass_scores, self.categories):
            queues[self.superclass_lookup[category]].append({
                                'index': index, # might be used when logits need to be queries from a different file
                                'label': self.superclass_lookup[label], 
                                'score': superclass_score})
        return queues
        
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
        self.superclass_scores = self.superclass_scores[shuffle_ids]
        
    def generate(self) -> Iterable[Tuple[int, int]]:
        for label, prediction in zip(self.labels, self.predictions):
            if label == prediction:
                entry = 0
            elif self.superclass_lookup[label] == self.superclass_lookup[prediction]:
                entry = 1
            else:
                entry = 2
            yield prediction, entry
            
    def group(self) -> None:
        return

    @classmethod
    def load_from_text(cls, dataset_name: str, superclass_lookup: Dict[int, int]) -> 'Dataset':
        fname = DATAFILE_DICT[dataset_name]
        array = np.genfromtxt(fname)
        labels = array[:, 0].astype(np.int)
        scores = array[:, 1:].astype(np.float)
        return cls(labels, scores, dataset_name, superclass_lookup)    
    
    @property
    def confusion_probs(self) -> np.ndarray:
        arr = confusion_matrix([self.superclass_lookup[i] for i in self.labels], 
                               [self.superclass_lookup[i] for i in self.predictions]).transpose()
        return arr / arr.sum(axis=-1, keepdims=True)


    @property
    def confusion_prior(self) -> np.ndarray:
        arr = np.zeros((self.num_groups, self.num_groups))
        for i in range(self.num_groups):
            arr[i] = self.superclass_scores[self.categories == i].sum(axis=0) / sum(self.categories == i)
        return arr