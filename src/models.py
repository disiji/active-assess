import copy
import math
from typing import List, Tuple, Any, Dict

import numpy as np
from scipy.stats import beta


class Model:
    """
    Abstract base class to be inhereted by all models.
    Derived classes must implement an update and sample method.
    """

    def update(self, predicted_class: int, true_class: int) -> None:
        """
        Update the model given a new observation.
        :param predicted_class: int
            The class predicted by the blackbox classifier.
        :param true_class: int
            The true class revealed by the oracle.
        """
        raise NotImplementedError

    def sample(self) -> np.ndarray:
        """
        Sample a parameter vector from the model posterior.
        :return: np.ndarray
            The sampled parameter vector.
        """
        raise NotImplementedError


class BetaBernoulli(Model):

    def __init__(self, k: int, prior=None, weight=None):
        self._k = k # number of arms
        self._prior = prior
        self._weight = weight
        if prior is None:
            self._prior = np.ones((k, 2)) * 0.5
        self._params = copy.deepcopy(self._prior)

    @property
    def mpe(self) -> np.ndarray:
        return self._params[:, 0] / (self._params[:, 0] + self._params[:, 1])

    @property
    def mle(self) -> np.ndarray:
        counts = self._params - self._prior + 0.0001
        return counts[:, 0] / (counts[:, 0] + counts[:, 1])

    @property
    def variance(self) -> np.ndarray:
        return beta.var(self._params[:, 0], self._params[:, 1])

    def get_params(self) -> np.ndarray:
        return self._params

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Draw sample thetas from the posterior.
        :param num_samples: int
            Number of times to sample from posterior. Default: 1.
        :return: An (k, num_samples) array of samples of theta. If num_samples == 1 then last dimension is squeezed.
        """
        theta = np.random.beta(self._params[:, 0], self._params[:, 1], size=(num_samples, self._k))
        return np.array(theta).T.squeeze()

    def update(self, category:int, labeled_datapoint: Dict[str, Any]) -> None:
        """
        Updates the posterior of the Beta-Bernoulli model with one labeled data point.
        """
        label = np.argmax(labeled_datapoint['score'])
        if label == labeled_datapoint['label']:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1
    
    def reward(self, reward_type) -> None:
        def r(theta_hat): # expected reward with sampled model state "theta"
            if reward_type == 'least_accurate':
                return -theta_hat # E(y|theta_hat)
            elif reward_type == 'most_accurate':
                return theta_hat
            elif reward_type == 'overall_accuracy':
                var_plus_1 = beta.var(self._params[:, 0]+1, self._params[:, 1])
                var_plus_0 = beta.var(self._params[:, 0], self._params[:, 1]+1)
                E_var =  var_plus_1 * theta_hat + var_plus_0 * (1-theta_hat)
                return (self.variance - E_var) * self._weight
        return r(self.sample())
    
            
class DirichletMultinomialCost(Model):
    """
    Multinomial w/ Dirichlet prior for predicted class cost estimation.
    WARNING: Arrays passed to constructor are copied!

    Parameters
    ==========
    alphas : np.ndarray
        An array of shape (n_classes, n_classes) where each row parameterizes a single Dirichlet distribution.
    costs : np.ndarray
        An array of shape (n_classes, n_classes). The cost matrix.
    """

    def __init__(self, alphas: np.ndarray, costs: np.ndarray) -> None:
        assert alphas.shape == costs.shape
        self._alphas = np.copy(alphas)
        self._costs = np.copy(costs)

    def update(self, predicted_class: int, true_class: int) -> None:
        """Update the posterior of the model."""
        self._alphas[predicted_class, true_class] += 1

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw sample expected costs from the posterior.
        :param n_samples: int
            Number of times to sample from posterior. Default: 1.
        :return: An (n, n_samples) array of expected costs. If n_samples == 1 then last dimension is squeezed.
        """
        # Draw multinomial probabilities (e.g. the confusion probabilities) from posterior
        if n_samples == 1:
            posterior_draw = np.zeros_like(self._alphas)
            for i, alpha in enumerate(self._alphas):
                posterior_draw[i] = np.random.dirichlet(alpha)
        else:
            posterior_draw = np.zeros((n_samples, *self._alphas.shape))
            for i, alpha in enumerate(self._alphas):
                posterior_draw[:, i, :] = np.random.dirichlet(alpha, size=(n_samples,))

        # Compute expected costs of each predicted class
        expected_costs = (np.expand_dims(self._costs, 0) * posterior_draw).sum(axis=-1)
        return expected_costs.squeeze()

    def mpe(self) -> np.ndarray:
        """Mean posterior estimate of expected costs"""
        z = self._alphas.sum(axis=-1, keepdims=True)
        expected_probs = self._alphas / z
        expected_costs = (self._costs * expected_probs).sum(axis=-1)
        return expected_costs

    def confusion_matrix(self) -> np.ndarray:
        z = self._alphas.sum(axis=-1, keepdims=True)
        return self._alphas / z
