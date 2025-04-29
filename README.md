# Alter: Time Series Forecasting Models Capturing Spatial-Temporal-Random Information from Fusion of Time index and Multivariate value

<p align="center">
<img src=".\pics\Alter.jpg" width = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> The overall architecture of Alter model.
</p>



Official PyTorch code repository for the Alter.

* To overcome the shortcomings of time-index models and multivariate-value models, we propose a new input that fuses time index and multivariate value.
* In order to dominate the fusion input, a closed Ridge Regressor based on meta-learning mode is used to construct the Alter, and the multiscale Gaussian random Fourier mapping feature method and Spatial-Temporal-Random MLP concatenation learning are used to learn their feature information.
* Under six benchmarks for long-term time series prediction, including ETTm2, ECL, Exchange, Traffic, Weather and ILI, Alter achieved an average performance improvement of 21.5%. Among them, the Alter based on fusion of time index and multivariate value has the best performance in the Alter series.

## Spatial-Temporal-Random MLP (STRMLP)

Inspired by many MLP precedents of joint learning information, our proposed STRMLP can capture spatial correlation, temporal dependence and randomness information of sequences.

<p align="center">
<img src=".\pics\STRMLP.jpg" width = "800" alt="" align=center />
<br><br>
<b>Figure 2.</b> The structure of each layer of STRINR in STRMLP.
</p>

## Requirements

Dependencies for this project can be installed by:

```bash
pip install -r requirements.txt
```

## Quick Start

### Data

To get started, you will need to download the datasets as described in our paper:

* Pre-processed datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing), as obtained from [Autoformer's](https://github.com/thuml/Autoformer) GitHub repository.
* Place the downloaded datasets into the `storage/datasets/` folder, e.g. `storage/datasets/ETT-small/ETTm2.csv`.

### Reproducing Experiment Results

We provide some scripts to quickly reproduce the results reported in our paper. There are two options, to run the full
hyperparameter search, or to directly run the experiments with hyperparameters provided in the configuration files.

__Option A__: Run the full hyperparameter search.

1. Run the following command to generate the experiments: `make build-all path=experiments/configs/hp_search`.
2. Run the following script to perform training and evaluation: `./run_hp_search.sh` (you may need to
   run `chmod u+x run_hp_search.sh` first).

__Option B__: Directly run the experiments with hyperparameters provided in the configuration files.

1. Run the following command to generate the experiments: `make build-all path=experiments/configs/ETTm2`.
2. Run the following script to perform training and evaluation: `./run.sh` (you may need to run `chmod u+x run.sh`
   first).

Finally, results can be viewed on tensorboard by running `tensorboard --logdir storage/experiments/`, or in
the `storage/experiments/experiment_name/metrics.npy` file.

## Main Results

We introduce six real-world datasets, covering five major long-term series forecasting application areas: energy, traffic, economy, weather and disease. Then test Alter on these datasets and show that Alter performs better, achieving a relative improvement of 21.5%.
<p align="center">
<img src=".\pics\results.jpg" width = "700" alt="" align=center />
<br><br>
</p>


## Detailed Usage

Further details of the code repository can be found here. The codebase is structured to generate experiments from
a `.gin` configuration file based on the `build.variables_dict` argument.

1. First, build the experiment from a config file. We provide 2 ways to build an experiment.
    1. Build a single config file:
       ```
       make build config=experiments/configs/folder_name/file_name.gin
       ```
    2. Build a group of config files:
       ```bash
       make build-all path=experiments/configs/folder_name
       ```
2. Next, run the experiment using the following command
    ```bash 
    python -m experiments.forecast --config_path=storage/experiments/experiment_name/config.gin run
   ```
   Alternatively, the first step generates a command file found in `storage/experiments/experiment_name/command`, which
   you can use by the following command,
   ```bash
   make run command=storage/experiments/experiment_name/command
   ```
3. Finally, you can observe the results on tensorboard
   ```bash
   tensorboard --logdir storage/experiments/
   ```
   or view the `storage/experiments/experiment_name/metrics.npy` file.

## Acknowledgements

The implementation of Alter relies on the resources of the following code libraries and repositories, as well as code snippets from reference papers. We appreciate the original authors for opening up their work.

* https://github.com/thuml/Nonstationary_Transformers
* https://github.com/thuml/TimesNet
* https://github.com/salesforce/DeepTime
* https://github.com/MAZiqing/FEDformer
* https://github.com/thuml/Autoformer
* https://github.com/CV-ZhangXin/AKConv
* https://github.com/laohuu/deep_learning_implementations/blob/main/algorithms/lstm/lstm.py
* https://arxiv.org/abs/2402.10487

## Citation

Please consider citing if you find this code useful to your research.
<pre>
</pre>

## Contact

If you have any questions or want to use the code, please contact zhangshengbo2049@outlook.com.