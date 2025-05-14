import numpy as np

def stratified_sampling(n, k):
    # split range 0-n into k bins, sample uniformly from each bin
    # return a list of k values

    bin_size = n / k
    samples = []
    for i in range(k):
        #sample real valueuniformally from range bin_size * i to bin_size * (i + 1)
        samples.append(np.random.uniform(bin_size * i, bin_size * (i + 1)))
    return samples