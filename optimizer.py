from Genetic_optimization_241011 import runsimulation
from joblib import Parallel, delayed
from neuron_model_utility import compute_block_functions,print_figure
import time
t = time.time()

neuron_list=['19d03004_rita']#'19d03004_rita','19d03000_rita','neuron191113002_S75',]
num_cores = len(neuron_list)#multiprocessing.cpu_count();


Ltt = Parallel(n_jobs=num_cores, verbose=50)(delayed(runsimulation)(neuron_list[i]) for i in range(len(neuron_list)))

monod_active=True
for i in range(len(neuron_list)):
    compute_block_functions(neuron_list[i])
    print_figure(neuron_list[i],monod_active)
elapsed = time.time() - t
