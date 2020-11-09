All the algorithms described in the paper are implemented in the Python library `ntp`.

The code is written in Python version 3.7 and was tested on Ubuntu 18.04.
The code requires PyTorch version 1.5 with CUDA enabled.
Other requirements are listed in [`requirements.txt`](requirements.txt).

To install the library run
```
cd code/
pip install -e .
```

The datasets used in the paper can be found in `data/`.


Jupyter notebooks reproducing the experimental results can be found in the `notebooks/` folder:
- [Differentiable relaxation](notebooks/differentiable_relaxation.ipynb)
- [Scalability](notebooks/scalability.ipynb)
- [Density estimation](notebooks/density_estimation.ipynb)
- [Variational inference for MJPs](notebooks/variational_inference.ipynb)


You can also train the model using command line with
```
python scripts/experiment.py
```
To see the command line arguments, use
```
python scripts/experiment.py --help
```
