import abc
import numpy as np


class DPSelectionMechanism(object):
    """Main class for differentially private selection mechanisms."""

    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def select(self, k):
        """Performs differentially private top-k selection using the given mechanism.
        
        Args:
            k: Number of elements to select.
        """
        pass

    
    @staticmethod
    def privacy_budget_per_iteration(epsilon_prime, delta_prime, k, prec=0.0001):
        """Returns the epsilon budget per iteration for a sequential composition of k 
           exponential mechanisms with a total budget of epsilon_prime.
           From [Dong, Durfee and Rogers. Optimal differential privacy composition
                 for exponential mechanisms and the cost of adaptivity, 2019]
        
        Args: 
            epsilon_prime: Total privacy budget for the sequential composition.
            delta_prime: The delta budget specifically for the composition.
            k: Number of mechanisms in the composition.
            prec: Precision for the number approximation for the epsilon budget returned.
        
        Returns:
            A real number with the budget that each of the k mechanisms can use such that
                the sequential composition will have overall budget epsilon_prime.
        """
        eps_it = epsilon_prime/k
        t_min = 0
        while t_min <= epsilon_prime:
            eps_it = eps_it + prec
            t_1 = k*eps_it
            t_2 = k*( eps_it/(1-np.exp(-eps_it)) -1 - np.log(eps_it/(1-np.exp(-eps_it))) ) + eps_it*np.sqrt( (k/2)*(np.log(1/delta_prime)) )
            t_min = min(t_1, t_2)
        return eps_it-prec

    
    @staticmethod
    def delta_per_query(delta_u, multiplier=1, prec=0.0001):
        """Returns the delta_q for multiple queries such that the overall budget is delta_u.
        
        Args:
            delta_u: Total privacy budget for multiple queries.
            multiplier: Multiplier for the delta_max condition. May change for different mechanisms.
            prec: Precision for the number approximation for the delta budget returned.
            
        Returns:
            A real number with the delta budget that each query can use such that overall 
                the multiple queries use the total budget delta_u.
        
        """
        delta_q = delta_u
        delta_max = multiplier*(delta_q/4)*(3 + np.log(1/delta_q))
        while delta_max > delta_u:
            delta_q = delta_q*(1-prec)
            delta_max = multiplier*(delta_q/4)*(3 + np.log(1/delta_q))
        return delta_q