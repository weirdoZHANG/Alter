# Alter: A Deep Meta-Optimization Model with Learning Spatial-Temporal-Random Information for Time Series Forecasting
<p align="center">
  <img src=".\pics\Alter.png" width = "240" alt="" align=center />
  <img src=".\pics\Closed_Meta-optimization.png" width = "300" alt="" align=center />
  <br><br>
  <b>Figure 1.</b> The overall architecture of the Alter model (left). The closed meta-optimization framework (right).
</p>

Official PyTorch code repository for the Alter.

* Introducing the proposed Alter model and its crucial internal modules in detail.
* Designing key algorithms like Spatial AKConv, EGU, and Random INR to efficiently extract Spatial-Temporal-Random information, and using Standard INR to enhance the feature representation.
* Achieving state-of-the-art performance in various prediction tasks, Alter has verified the effectiveness of its key components through numerous ablation experiments.

## Spatial-Temporal-Random MLP (STRMLP)

Our proposed STRMLP can capture spatial correlation, temporal dependence and randomness information of sequences.

<p align="center">
<img src=".\pics\STRMLP.png" width = "800" alt="" align=center />
<br><br>
<b>Figure 2.</b> The structure of one layer of STRINR in STRMLP, which includes four types of algorithms: EGU, Spatial AKConv, Random INR, and Standard INR.
</p>

## Requirements

Dependencies for this project can be installed by:

```bash
pip install -r requirements.txt
```

## Quick Start

### Datasets

To get started, you will need to download the datasets as described in our paper:

* See `datasets/README.md`

### Reproducing Experiment Results

We provide some scripts to quickly reproduce the results reported in our paper. There are two options, to run the full
hyperparameter search, or to directly run the experiments with hyperparameters provided in the configuration files.

__Option A__: Run the configuration of TM_forecast, TS_forecast or FS_forecast.

1. Run the following command to generate the experiments (Take TM_forecast as an example): `make build-all path=configs/TM_forecast/*/*.gin`.
2. Run the following script to perform training and evaluation: `./run.sh` (you may need to run `chmod u+x run.sh` first).

__Option B__: Run the configuration of a certain dataset in TM_forecast, TS_forecast or FS_forecast.

1. Run the following command to generate the experiments (Take the ETTm2 dataset in TM_forecast as an example): `make build-all path=configs/TM_forecast/ETTm2/*.gin`.
2. Run the following script to perform training and evaluation: `./run.sh` (you may need to run `chmod u+x run.sh` first).

__Option C__: Run the configuration of a certain horizon length of a certain dataset in TM_forecast, TS_forecast or FS_forecast.

1. Run the following command to generate the experiments (Take the ETTm2_96TM.gin of the ETTm2 dataset in TM_forecast as an example): `make build-all path=configs/TM_forecast/ETTm2/ETTm2_96TM.gin`.
2. Run the following script to perform training and evaluation: `./run.sh` (you may need to run `chmod u+x run.sh` first).

## Main Results

Alter performs relatively evenly across multiple domains and tasks, demonstrating its potential advantages as a general-purpose model. The following are the main results of multivariate, univariate and few-shot prediction.

1. Multivariate prediction results.
<p align="center">
<img src=".\pics\multivariate.png" width = "700" alt="" align=center />
<br><br>
</p>

2. Univariate prediction results.
<p align="center">
<img src=".\pics\univariate.png" width = "700" alt="" align=center />
<br><br>
</p>

3. Few-shot prediction results.
<p align="center">
<img src=".\pics\few-shot.png" width = "700" alt="" align=center />
<br><br>
</p>

## Acknowledgements

The implementation of Alter relies on the resources of the following code libraries and repositories, as well as code snippets from reference papers. We appreciate the original authors for opening up their work.

* https://github.com/thuml/TimesNet
* https://github.com/salesforce/DeepTime
* https://github.com/thuml/Autoformer
* https://github.com/CV-ZhangXin/AKConv

## Citation

Please consider citing if you find this code useful to your research.
<pre>
</pre>

## Contact

If you have any questions or want to use the code, please contact `zhangshengbo2049@outlook.com`.
