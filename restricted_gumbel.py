import numpy as np

from dp_selection_mechanism import DPSelectionMechanism
from limit_domain import LimitDomain


class RestrictedGumbel(DPSelectionMechanism):
    """
        Implements the RestrictedGumbel (RG) mechanism for top-k selection.
    """
    
    def __init__(self, score, epsilon_total, delta_total):
        """Initializes the RestrictedGumbel mechanism.
        
        Args: 
            score: Array with scores of elements IN DESCEDING ORDER, for AT LEAST (kbar+1) elements,
                as the mechanism will access the elements with top-kbar scores and the element with
                the top-(kbar+1) score is used for the threshold.
            epsilon_total: Total epsilon privacy budget for the top-k selection.
            delta_total: Total epsilon privacy budget for the top-k selection.
        """
        self.score = score
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        
        self.delta_q = RestrictedGumbel.delta_per_query(delta_total/2)

        
    def select(self, k, kbar):
        """Performs differentially private top-k selection using RestrictedGumbel (RG).
        
        Args:
            k: Number of elements to select.
            kbar: Number of elements to consider for selection.
            
        Returns:
            Array of size at most k, containing indices in {0, 1, ..., kbar-1} 
                representing the elements selected by RestrictedGumbel (RG).
        """
        
        # Half of delta_total goes to delta_M of composition, other half goes to delta_U of RG
        delta_half = (self.delta_total)/2
        
        # Privacy budget per iteration
        epsilon_per_it_for_ld = LimitDomain.privacy_budget_per_iteration(self.epsilon_total, delta_half, k)
        epsilon_t = 4*epsilon_per_it_for_ld
        remaining_epsilon = self.epsilon_total - epsilon_t
        epsilon_per_it_for_rg = RestrictedGumbel.privacy_budget_per_iteration(remaining_epsilon, delta_half, k)
        
        # Elements (scores, indices) considered: Only kbar
        hst_rg = self.score[:kbar]
        ind_rg = np.array(list(range(kbar)))

        # Add Gumbel noise for selection
        if epsilon_per_it_for_rg > 0:
            hst_ns = hst_rg + np.random.gumbel(0, 1/epsilon_per_it_for_rg, kbar)
        else:
            hst_ns = hst_rg + np.random.gumbel(0, 1/0.00000000000001, kbar)

        # Sort noisy scores
        noisy_sort_order = hst_ns.argsort()[::-1]
        ind_ns = ind_rg[noisy_sort_order] # Sort indices to get real index

        # Select k elements
        topk_indic = ind_ns[:k]
        topk_score = hst_rg[topk_indic] # Get original scores, not with gumbel noise

        # Threshold
        threshold = np.log(1/self.delta_q)/(epsilon_t/2) + np.random.laplace(0, 1/(epsilon_t/2), 1)

        # Noisy scores: filt_ld[kbar] is the (kbar+1)-th element
        noise_lap = np.random.laplace(0, 1/(epsilon_t/2), k)
        noisy_scores = topk_score - self.score[kbar] - 1 + noise_lap

        noisy_scores = np.append(noisy_scores, -np.inf)

        good_score = np.argmin( noisy_scores > threshold )
        ind_res = topk_indic[:good_score]

        return ind_res
    