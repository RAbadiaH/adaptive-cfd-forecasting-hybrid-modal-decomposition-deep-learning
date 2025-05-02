import scipy.io
import mat73

import numpy as np

def load_data(case):

    print("\nLoading Data ...", end='\r')
    
    if (case == 'laminar'):
        f = mat73.loadmat('./Datasets/laminar_flow.mat')
        data_ten = f['Tensor']
        data_ten = data_ten[..., 100:]

    elif (case == 'JetLES'):
        f = scipy.io.loadmat('./Datasets/jetLES.mat')
        data_ten = f.get('p')
        data_ten = np.transpose(data_ten, [1,2,0])
        data_ten = data_ten[None, ..., :2000]
    
    elif (case == 'turbulent'):
        f = mat73.loadmat('./Datasets/turbulent_flow.mat')
        data_ten = f.get('Tensor')
        data_ten = data_ten[..., :2000]

    else:

        raise TypeError("Dataset not found, check load_data.py.")

    print("Loading Data: DONE\n")

    print("\n")

    return data_ten
