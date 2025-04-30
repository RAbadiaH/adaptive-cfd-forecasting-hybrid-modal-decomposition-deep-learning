# adaptive-cfd-forecasting-hybrid-modal-decomposition-deep-learning
An adaptive, data-driven framework for stable and efficient forecasting in CFD using a hybrid model combining modal decomposition and deep learning.

![Adaptive framework combining precomputed data from CFD with a hybrid POD-DL model.](https://github.com/user-attachments/assets/302acc9d-087a-4299-a066-3396aeaa5912)

This repository contains the official implementation of the paper **An Adaptive Framework for Autoregressive Forecasting in CFD Using Hybrid Modal Decomposition and Deep Learning**.

Abstract:
This work introduces, for the first time to the authors’ knowledge, a proof of concept for a fully data‑driven adaptive framework that stabilizes deep learning autoregressive forecasting models over long horizons to accelerate computational fluid dynamics (CFD) simulations. By training on precomputed CFD data, our approach predicts flow evolution for a given interval, then new CFD data are generated to update the model, ensuring stability over extended time horizons. Tested across three flow scenarios, from laminar to turbulent regimes, this methodology significantly reduces computational cost, of up to $80\%$, while preserving physics‐consistent accuracy. Its purely data‑driven nature makes it readily extensible to a wide range of applications. This method will be incorporated into ModelFLOWs-app’s next version release.

## Replication of Results
To replicate results from the paper, [download](https://drive.google.com/drive/folders/1_MkWVuWWoE3hGKPT0FbCba234KJ06kQo) the dataset of the laminar flow (Tensor.mat), and specify the path in `load_data.py`. We do not have the permissions to share the other datasets, but the dataset for the turbulent flow can be found [here](https://github.com/mendezVKI/MODULO/tree/master/download_all_data_exercises).

To custimize the adaptive process, tweak the hyperparameters defined in the script `main.py`.

We highly recommend to create a [python enviroment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) and intall the libraries defined in `requirements.txt` to avoid possible errors due to versions conflict.
