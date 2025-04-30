import scipy.io
import mat73

import numpy as np

def load_data(case):

    print("\nLoading Data ...", end='\r')
    
    if (case == 'cil3D'):
        f = mat73.loadmat('/home/rodrigo/Desktop/svd_nn_adaptive/Tensor.mat')
        data_ten = f['Tensor']
        data_ten = data_ten[..., 100:]

    elif (case == 'cil2D'):
        f = scipy.io.loadmat('/home/rodrigo/Downloads/Bases_de_datos_LCSVD/Tensor_cylinder_Re100.mat')
        data_ten = f['Tensor']

    elif (case == 'JetLES'):
        f = scipy.io.loadmat('./Datasets/jetLES.mat')
        data_ten = f.get('p')
        data_ten = np.transpose(data_ten, [1,2,0])
        data_ten = data_ten[None, ..., :2000]
    
    elif (case == 'vki'):
        f = mat73.loadmat('/home/rodrigo/Downloads/Bases_de_datos_LCSVD/VKI_Re4000.mat')
        data_ten = f.get('Tensor')
        data_ten = data_ten[..., :2000]

    elif (case == 'synthetic'):
        f = mat73.loadmat('/home/rodrigo/Documents/Rodrigo/SyntheticJet/TensorMatrix_100x70.mat')
        data_ten = f['TensorMatrix'][..., :2000]

    else:

        raise TypeError("The case has to be 'cil3D' or 'JetLES'.")

    print("Loading Data: DONE\n")

    print("\n")

    return data_ten