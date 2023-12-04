import numpy as np

from dp_selection_mechanism import DPSelectionMechanism


class LimitDomain(DPSelectionMechanism):
    """
        Implements the LimitDomain (LD) mechanism for top-k selection 
        of [Durfee and Rogers, NeurIPS 2019].
    """
    
    
    def __init__(self, score, epsilon_total, delta_total):
        """Initializes the LimitDomain mechanism.
        
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

        
    def select(self, k, kbar):
        """Performs differentially private top-k selection using LimitDomain (LD).
        
        Args: 
            k: Number of elements to select.
            kbar: Number of elements to consider for selection.
            
        Returns:
            Array of size at most k, containing indices in {0, 1, ..., kbar-1} 
                representing the elements selected by LimitDomain (LD).
        """
        
        # Half of delta_total goes to delta_prime of composition, other half goes to delta of LD
        delta_half = (self.delta_total)/2
        
        # Privacy budget per iteration
        epsilon_per_it = LimitDomain.privacy_budget_per_iteration(self.epsilon_total, delta_half, k)

        # Elements (scores, indices) considered: Only kbar
        hst_ld = self.score[:kbar]
        ind_ld = np.array(list(range(kbar)))

        # Bot: count of element with indice (kbar+1) plus other two terms
        bot_ct = self.score[kbar] + 1 + np.log(kbar/delta_half)/epsilon_per_it

        # Add Gumbel noise for selection
        hst_ns = hst_ld + np.random.gumbel(0, 1/epsilon_per_it, kbar)
        bot_ns = bot_ct + np.random.gumbel(0, 1/epsilon_per_it, 1)

        # Sort noisy counts
        noisy_sort_order = hst_ns.argsort()[::-1]
        ind_ns = ind_ld[noisy_sort_order] # Sort indices too, to get correct index
        hst_ns = hst_ns[noisy_sort_order]

        # Only get indices of those with count above Bot-Noisy-Count
        ind_ns = ind_ns[hst_ns > bot_ns]

        # Return at most k elements
        ind_ns = ind_ns[:k]

        return ind_ns
