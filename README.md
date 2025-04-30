# adaptive-cfd-forecasting-hybrid-modal-decomposition-deep-learning
An adaptive, data-driven framework for stable and efficient forecasting in CFD using a hybrid model combining modal decomposition and deep learning.

This repository contains the official implementation of the paper "An Adaptive Framework for Autoregressive Forecasting in CFD Using Hybrid Modal Decomposition and Deep Learning".

Abstract:
This work introduces, for the first time to the authors’ knowledge, a proof of concept for a fully data‑driven adaptive framework that stabilizes deep learning autoregressive forecasting models over long horizons to accelerate computational fluid dynamics (CFD) simulations. By training on precomputed CFD data, our approach predicts flow evolution for a given interval, then new CFD data are generated to update the model, ensuring stability over extended time horizons. Tested across three flow scenarios, from laminar to turbulent regimes, this methodology significantly reduces computational cost, of up to $80\%$, while preserving physics‐consistent accuracy. Its purely data‑driven nature makes it readily extensible to a wide range of applications. This method will be incorporated into ModelFLOWs-app’s next version release.
