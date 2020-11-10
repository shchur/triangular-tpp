# Fast and Flexible Temporal Point Processes with Triangular Maps
<p align="center">
  <img width="640" src="https://i.postimg.cc/Y9R38vt6/tritpp-logo.png">
</p>

This repository includes a reference implementation of the algorithms described in ["Fast and Flexible Temporal Point Processes with Triangular Maps"](https://arxiv.org/abs/2006.12631) by Oleksandr Shchur, Nicholas Gao, Marin Biloš and Stephan Günnemann (Oral, NeurIPS 2020).

Temporal point processes (TPPs) allow us to define probability distributions over variable-length event sequences in some time interval `[0, t_max]`.
In our paper, we show how to define TPPs using invertible transformations, similar to normalizing flows.
The code includes new parametrizations for several existing TPPs as well as a new, more flexible model.
Our parametrizations allow to both draw samples & compute likelihood in parallel, which leads to signficant speedups compared to traditional RNN-based models.

The following models are available in [`ttpp.models`](https://github.com/shchur/triangular-tpp/blob/main/ttpp/models.py):
- `Inhomogeneous Poisson Process`
- `Renewal Process`
- `Modulated Renewal Process`
- `TriTPP`
- `Autoregressive` (RNN-based TPP with slow sampling)

## Requirements

The code is written in Python version 3.7 and was tested on Ubuntu 18.04.
The code requires PyTorch version 1.5 with CUDA enabled.
Other requirements are listed in [`requirements.txt`](requirements.txt).

To install the library run
```
pip install -e .
```
The datasets used in the paper can be found in `data/`.

## Usage
Jupyter notebooks reproducing the experimental results can be found in the `notebooks/` folder:
- [Differentiable relaxation](notebooks/differentiable_relaxation.ipynb)
- [Scalability](notebooks/scalability.ipynb)
- [Density estimation](notebooks/density_estimation.ipynb)
- [Variational inference for MJPs](notebooks/variational_inference.ipynb)


You can also train the model using command line. For example, to train the `TriTPP` model on the `taxi` dataset run
```
python scripts/experiment.py taxi TriTPP
```
To see the command line arguments, use
```
python scripts/experiment.py --help
```

## Cite
Please cite our paper if you use this code or data in your own work:
```
@inproceedings{shchur2020fast,
  title = {Fast and Flexible Temporal Point Processes with Triangular Maps},
  author = {Shchur, Oleksandr and Gao, Nicholas and Bilo\v{s}, Marin and G{\"u}nnemann, Stephan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020} 
}
```
