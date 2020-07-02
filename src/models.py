import copy
import math
from typing import List, Tuple, Any, Dict

import numpy as np
from scipy.stats import beta, dirichlet


class Model:
    """
    Abstract base class to be inhereted by all models.
    Derived classes must implement an update and sample method.
    """

    def update(self, category:int, labeled_datapoint: Dict[str, Any]) -> None:
        """
        Updates the posterior of the model with one labeled data point.
        labeled_datapoint: index, label, score
        """
        raise NotImplementedError

    def sample(self) -> np.ndarray:
        """
        Sample a parameter vector from the model posterior.
        :return: np.ndarray
            The sampled parameter vector.
        """
        raise NotImplementedError
        
    def reward(self) -> np.ndarray:
        """
        Compute the reward with current model parameters.
        :return: np.ndarray
            The reward of each arm.
        """


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
        label = np.argmax(labeled_datapoint['score'])
        if label == labeled_datapoint['label']:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1
    
    def reward(self, reward_type, group0=None, group1=None) -> np.ndarray:
        # group0 and group1 only need to specified when reward t
        def r(theta_hat): # expected reward with sampled model state "theta"
            if reward_type == 'least_accurate':
                return -theta_hat # E(y|theta_hat)
            
            elif reward_type == 'most_accurate':
                return theta_hat
            
            elif reward_type == 'groupwise_accuracy':
                var_plus_1 = beta.var(self._params[:, 0]+1, self._params[:, 1])
                var_plus_0 = beta.var(self._params[:, 0], self._params[:, 1]+1)
                E_var =  var_plus_1 * theta_hat + var_plus_0 * (1-theta_hat)
                return (self.variance - E_var) * self._weight ** 2
            
            elif reward_type == 'difference':
                if group0 is None or group1 is None:
                    raise ValueError
                
                def rope(alpha0, alpha1, beta0, beta1):
                    num_samples = 10000
                    theta_0 = np.random.beta(alpha0, beta0, size=(num_samples))
                    theta_1 = np.random.beta(alpha1, beta1, size=(num_samples))
                    delta = theta_0 - theta_1
                    return max((delta < -0.05).mean(), 
                               (np.abs(delta) <= 0.05).mean(), 
                               (delta > 0.05).mean())
                
                alpha0, beta0 = self._params[group0]
                alpha1, beta1 = self._params[group1]
                rope_plus_1 = np.array([rope(alpha0+1, alpha1, beta0, beta1),
                                        rope(alpha0, alpha1+1, beta0, beta1)])
                rope_plus_0 = np.array([rope(alpha0, alpha1, beta0+1, beta1),
                                        rope(alpha0, alpha1, beta0, beta1+1)])
                theta_hat = theta_hat[[group0, group1]]
                E_rope = rope_plus_1 * theta_hat + rope_plus_0 * (1-theta_hat)
                r = np.ones(self._k) * (-1) # set reward of other groups to be -1
                r[[group0, group1]] = E_rope
                return r
                
        return r(self.sample())
    
            
class DirichletMultinomial(Model):
    """
    Multinomial w/ Dirichlet prior for predicted class cost estimation.
    WARNING: Arrays passed to constructor are copied!

    Parameters
    ==========
    prior : np.ndarray
        An array of shape (n_classes, n_classes) where each row parameterizes a single Dirichlet distribution.
    costs : np.ndarray
        An array of shape (n_classes, n_classes). The cost matrix.
    weight: np.ndarray
        An array of
    """

    def __init__(self, prior: np.ndarray, costs: np.ndarray, weight=None) -> None:
        assert prior.shape == costs.shape
        self._k = prior.shape[0]
        self._prior = prior
        self._alphas = np.copy(prior)
        self._costs = np.copy(costs)
        self._weight = weight

    def update(self, category:int, labeled_datapoint: Dict[str, Any]) -> None:
        label = np.argmax(labeled_datapoint['score'])
        self._alphas[category, label] += 1

    def sample_cost(self, n_samples: int = 1) -> np.ndarray:
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
    

    def sample(self, n_samples: int = 1) -> np.ndarray:
        if n_samples == 1:
            posterior_draw = np.zeros_like(self._alphas)
            for i, alpha in enumerate(self._alphas):
                posterior_draw[i] = np.random.dirichlet(alpha)
        else:
            posterior_draw = np.zeros((n_samples, *self._alphas.shape))
            for i, alpha in enumerate(self._alphas):
                posterior_draw[:, i, :] = np.random.dirichlet(alpha, size=(n_samples,))
        return posterior_draw
    
    @property
    def mpe_cost(self) -> np.ndarray:
        """Mean posterior estimate of expected costs"""
        return (self._costs * self.mpe).sum(axis=-1)
    
    @property
    def mpe(self) -> np.ndarray:
        return self._alphas / self._alphas.sum(axis=-1, keepdims=True)
    
    @property
    def mle(self) -> np.ndarray:
        params = (self._alphas - self._prior)
        z = params.sum(axis=-1, keepdims=True)
        return params / z        
    
    @property
    def entropy(self) -> np.ndarray:
        return np.array([dirichlet.entropy(self._alphas[i] + 1e-6) for i in range(self._k)])
    
    def reward(self, reward_type, group0=None, group1=None) -> np.ndarray:
        def r(alpha_hat):
            if reward_type == 'confusion_matrix':
                new_entropy = np.zeros(alpha_hat.shape)
                for j in range(self._k): # true class
                    params = np.copy(alpha_hat)
                    params[:,j] += 1
                    new_entropy[:,j] = np.array([dirichlet.entropy(params[i] + 1e-6) for i in range(self._k)])
                E_entropy =  (new_entropy * alpha_hat).sum(axis=-1)
                return (self.entropy - E_entropy) * self._weight 
        return r(self.sample())
