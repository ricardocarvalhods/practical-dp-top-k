import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, pre_process_data

from restricted_gumbel import RestrictedGumbel
from limit_domain import LimitDomain


def experiments(score, dataset_nr, metric, nr_trials, epsilon_total_list, delta_total, k_list, kbar_multiplier):
    """Plots images with output size comparison for RG versus LD on various settings.
    
    Args:
       score: Dataset with scores.
       
       dataset_nr: Number of reference of dataset, i.e. Yelp = 1, Foursquare = 2, Gowalla = 3.
       
       metric: Name of utility metric to evaluate, i.e. 'sqr' or 'ner'.
      
       nr_trials: Number of times each individual setting will be independtly run.
       
       epsilon_total_list: List of values of epsilon_total for the total privacy budget. 
          Mechanisms will be (epsilon_total, delta_total)-DP.
       
       delta_total: Value of delta_total on the total privacy budget. 
          Mechanisms will be (epsilon_total, delta_total)-DP.
       
       k_list: List of values of k to perform top-k selection.
       
       kbar_multiplier: List of numbers to multiply k in order to define the value of kbar.
          For example, a value of 10 defines kbar = 10*k. Should be of same size than
          the k_list, as they are matched during the execution.
    """
    print("Starting {}...".format(metric))
    
    # Setup of names and colors for each kbar setting
    kbar_names = [str(n)+"k" if n > 1 else "k" for n in kbar_multiplier]
    kbar_colors = ['red', 'green', 'blue']

    for k in k_list:
        # Create plot for current k
        fig, ax = plt.subplots()
        
        true_topk_scores = np.sum(score[:k])

        for kbar_i in range(len(kbar_multiplier)):
            kbar = k*kbar_multiplier[kbar_i]

            AVG_RG_per_eps = []
            AVG_LD_per_eps = []

            STD_RG_per_eps = []
            STD_LD_per_eps = []
            
            def sum_of_scores(output, score):
                return np.sum([score[elem] for elem in output])

            for epsilon_total in epsilon_total_list:
                RG = RestrictedGumbel(score, epsilon_total, delta_total)
                LD = LimitDomain(score, epsilon_total, delta_total)

                if metric == 'sqr':
                    RG_output = [sum_of_scores(RG.select(k, kbar), score)/true_topk_scores for trial in range(nr_trials)]
                    AVG_RG_per_eps.append(np.mean(RG_output))
                    STD_RG_per_eps.append(np.std(RG_output))

                    LD_output = [sum_of_scores(LD.select(k, kbar), score)/true_topk_scores for trial in range(nr_trials)]
                    AVG_LD_per_eps.append(np.mean(LD_output))
                    STD_LD_per_eps.append(np.std(LD_output))
                elif metric == 'ner':
                    RG_output = [len(RG.select(k, kbar)) for trial in range(nr_trials)]
                    AVG_RG_per_eps.append(np.mean(RG_output))
                    STD_RG_per_eps.append(np.std(RG_output))

                    LD_output = [len(LD.select(k, kbar)) for trial in range(nr_trials)]
                    AVG_LD_per_eps.append(np.mean(LD_output))
                    STD_LD_per_eps.append(np.std(LD_output))
                else:
                    print("METRIC unknown")
                    return

            # Plot results for current kbar
            eb1 = ax.errorbar(epsilon_total_list, AVG_RG_per_eps, yerr=STD_RG_per_eps, 
                              label="RG: " + r"$\bar{k}$=" + str(kbar_names[kbar_i]), 
                              color=kbar_colors[kbar_i])
            eb1[-1][0].set_alpha(0.4)

            eb2 = ax.errorbar(epsilon_total_list, AVG_LD_per_eps, yerr=STD_LD_per_eps, 
                              label="LD: " + r"$\bar{k}$=" + str(kbar_names[kbar_i]),
                              color=kbar_colors[kbar_i], linestyle='dotted')
            eb2[-1][0].set_linestyle('dotted')
            eb2[-1][0].set_alpha(0.4)

            if metric == 'sqr':
                plt.yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
                plt.xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
            else:
                if k == 100:
                    plt.yticks(np.array([0,20,40,60,80,99]))

        # Finish and show plot for current k and all kbars
        if metric == 'sqr':
            ax.set(xlabel=r"$\varepsilon_{total}$", ylabel=r"$\mathcal{SR}$ (Score Ratio)", 
                   title="RG versus LD: k={0} on {1}".format(k, r"$D_{" + str(dataset_nr) +"}$"))
        elif metric == 'ner':
            ax.set(xlabel=r"$\varepsilon_{total}$", ylabel='Number of elements returned', 
               title="RG versus LD: k={0} on {1}".format(k, r"$D_{" + str(dataset_nr) +"}$"))

        ax.grid()
        leg = ax.legend()
        
        if metric == 'sqr':
            fig.savefig("images/RG_LD_score_quality_{}-D{}.pdf".format(k, dataset_nr), dpi=600, bbox_inches='tight', pad_inches=0)
        elif metric == 'ner':
            fig.savefig("images/RG_LD_output_size_{0}-D{1}.pdf".format(k, dataset_nr), dpi=600, bbox_inches='tight', pad_inches=0)
        #plt.show()


def run_experiment(dataset_nr, nr_trials, epsilon_total_list, k_list, kbar_multiplier):
    if dataset_nr == 1:
        df = load_data("datasets/yelp_academic_dataset_tip.json", "yelp")
    elif dataset_nr == 2:
        df = load_data(["datasets/dataset_TSMC2014_NYC.txt", "datasets/dataset_TSMC2014_TKY.txt"], "foursquare")
    elif dataset_nr == 3:
        df = load_data("datasets/loc-gowalla_totalCheckins.txt", "gowalla")
    else:
        print("INVALID dataset number")
        return

    # Pre-process data
    nr_users, nr_elements, score = pre_process_data(df)

    print("TOTAL NUMBER OF USERS:", nr_users)
    print("TOTAL NUMBER OF ELEMENTS:", nr_elements)
    
    delta_total = 1/(2*nr_users)
    
    # Images generated are also saved on the 'images' folder
    experiments(score, dataset_nr, 'sqr', nr_trials, epsilon_total_list, delta_total, k_list, kbar_multiplier)
    experiments(score, dataset_nr, 'ner', nr_trials, epsilon_total_list, delta_total, k_list, kbar_multiplier)
    


params = {'legend.fontsize': 'large',
          'font.family':'serif',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


epsilon_total_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]

nr_trials = 100

k_list = [10, 50, 100]
kbar_multiplier = [1, 10, 50]


print("D1 - YELP")
run_experiment(1, nr_trials, epsilon_total_list, k_list, kbar_multiplier)

print("D2 - FOURSQUARE")
run_experiment(2, nr_trials, epsilon_total_list, k_list, kbar_multiplier)
    
print("D3 - GOWALLA")
run_experiment(3, nr_trials, epsilon_total_list, k_list, kbar_multiplier)

print("DONE")
