import logging
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from models import *

logger = logging.getLogger(__name__)

np.random.seed(0)
NUM_BINS = 20

############################################################################
"""
Update DATA_DIR, RESULTS_DIR, FIGURE_DIR
"""
# INTPUT FILES
DATA_DIR = '/home/disij/projects/bayesian-blackbox/data/'

DATAFILE_DICT = {
    'cifar100': DATA_DIR + 'cifar100/cifar100_predictions_dropout.txt',
    'imagenet': DATA_DIR + 'imagenet/resnet152_imagenet_outputs.txt',
    'imagenet2_topimages': DATA_DIR + 'imagenet/resnet152_imagenetv2_topimages_outputs.txt',
    '20newsgroup': DATA_DIR + '20newsgroup/bert_20_newsgroups_outputs.txt',
    'svhn': DATA_DIR + 'svhn/svhn_predictions.txt',
    'dbpedia': DATA_DIR + 'dbpedia/bert_dbpedia_outputs.txt',
}
LOGITSFILE_DICT = {
    'cifar100': DATA_DIR + 'cifar100/resnet110_cifar100_logits.txt',
    'imagenet': DATA_DIR + 'imagenet/resnet152_imagenet_logits.txt',
}

COST_MATRIX_FILE_DICT = {
    'human': DATA_DIR + 'cost/cifar100_people_full/costs.npy',
    'superclass': DATA_DIR + 'cost/cifar100_superclass_full/costs.npy'
}
COST_INFORMED_PRIOR_FILE = DATA_DIR + 'cost/cifar100_superclass_full/informed_prior.npy'


# OUTPUT FILES
RESULTS_DIR = '/home/disij/projects/active-assess/output/'
FIGURE_DIR = '../../figures/'

############################################################################
# DATA INFO
DATASET_LIST = ['imagenet', 'dbpedia', 'cifar100', '20newsgroup', 'svhn', 'imagenet2_topimages']
DATASIZE_DICT = {
    'cifar100': 10000,
    'imagenet': 50000,
    'imagenet2_topimages': 10000,
    '20newsgroup': 7532,
    'svhn': 26032,
    'dbpedia': 70000,
}
NUM_CLASSES_DICT = {
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet2_topimages': 1000,
    '20newsgroup': 20,
    'svhn': 10,
    'dbpedia': 14,
}

# PLOT
DATASET_NAMES = {
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
    'svhn': 'SVHN',
    '20newsgroup': '20 Newsgroups',
    'dbpedia': 'DBpedia',
}
TOPK_DICT = {'cifar100': 10,
             'imagenet': 10,
             'svhn': 3,
             '20newsgroup': 3,
             'dbpedia': 3}
EVAL_METRIC_NAMES = {
    'avg_num_agreement': '#agreements',
    'mrr': 'MRR'
}
############################################################################
# CIFAR100 meta data needed to map classes to superclasses and vice versa.

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
    "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle",
    "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
    "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]

CIFAR100_SUPERCLASSES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates", "people",
    "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2"
]

CIFAR100_REVERSE_SUPERCLASS_LOOKUP = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
}

CIFAR100_SUPERCLASS_LOOKUP = {class_: superclass for superclass, class_list in
                              CIFAR100_REVERSE_SUPERCLASS_LOOKUP.items() for class_ in
                              class_list}


############################################################################
import warnings
warnings.filterwarnings("ignore")

def eval_ece(confidences: List[float], observations: List[bool], num_bins=NUM_BINS):
    """
    Evaluate ECE given a list of samples with equal-width binning.
    :param confidences: List[float]
        A list of prediction scores.
    :param observations: List[bool]
        A list of boolean observations.
    :param num_bins: int
        The number of bins used to estimate ECE. Default: 10
    :return: float
    """
    confidences = np.array(confidences)
    observations = np.array(observations) * 1.0
    bins = np.linspace(0, 1, num_bins + 1)
    digitized = np.digitize(confidences, bins[1:-1])

    w = np.array([(digitized == i).sum() for i in range(num_bins)])
    w = w / sum(w)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        confidence_bins = np.array([confidences[digitized == i].mean() for i in range(num_bins)])
        accuracy_bins = np.array([observations[digitized == i].mean() for i in range(num_bins)])
    confidence_bins[np.isnan(confidence_bins)] = 0
    accuracy_bins[np.isnan(accuracy_bins)] = 0
    diff = np.absolute(confidence_bins - accuracy_bins)
    ece = np.inner(diff, w)
    return ece