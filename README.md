# adaptive-cfd-forecasting-hybrid-modal-decomposition-deep-learning
An adaptive, data-driven framework for stable and efficient forecasting in CFD using a hybrid model combining modal decomposition and deep learning.

![Adaptive framework combining precomputed data from CFD with a hybrid POD-DL model.](https://github.com/user-attachments/assets/302acc9d-087a-4299-a066-3396aeaa5912)

This repository contains the official implementation of the paper **An Adaptive Framework for Autoregressive Forecasting in CFD Using Hybrid Modal Decomposition and Deep Learning**.

Abstract:
This work introduces, for the first time to the authors’ knowledge, a proof of concept for a fully data‑driven adaptive framework that stabilizes deep learning autoregressive forecasting models over long horizons to accelerate computational fluid dynamics (CFD) simulations. By training on precomputed CFD data, our approach predicts flow evolution for a given interval, then new CFD data are generated to update the model, ensuring stability over extended time horizons. Tested across three flow scenarios, from laminar to turbulent regimes, this methodology significantly reduces computational cost, of up to $80\%$, while preserving physics‐consistent accuracy. Its purely data‑driven nature makes it readily extensible to a wide range of applications. This method will be incorporated into ModelFLOWs-app’s next version release.

## Replication of Results
> We strongly recommend setting up a [Python virtual enviroment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) and installing the required libraries listed in the `requirements.txt` file to prevent potential errors arising from version conflicts.

To replicate results from the paper, first download the datasets used in this work.
* The dataset corresponding to the laminar flow can be accessed [here](https://drive.google.com/drive/folders/1_MkWVuWWoE3hGKPT0FbCba234KJ06kQo) (Tensor.mat)
* The dataset corresponding to the turbulent flow is available [here](https://github.com/mendezVKI/MODULO/tree/master/download_all_data_exercises).

After downloading, specify the path in `load_data.py`.

### Running the adaptive framework
To custimize the adaptive process, tweak the hyperparameters defined in the script `main.py`. Specifically the one shown below,
* dfd
