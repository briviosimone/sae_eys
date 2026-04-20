# Deep symmetric autoencoders from the EYS perspective
This is the repository accompanying the paper

*S. Brivio, N. R. Franco, [Deep symmetric autoencoders from the Eckart-Young-Schmidt perspective](https://arxiv.org/abs/2506.11641) (2025)*,

providing *(i)* a novel mathematical framework for symmetric autoencoders, *(ii)* suitable error estimates, and
*(iii)* a brand-new data-driven initialization strategy.

### Installation
We suggest to install the library dependecies in a clean conda environment, namely, using either
~~~bash
conda create -n sym-ae python=3.11.9
conda activate sym-ae
conda install -c conda-forge fenics
~~~
Then, after cloning the library, one can install the repo with pip (with ```pip install -e .```). Before running the scripts, to verify the correctness of the installation, please run ```pytest``` from the main folder. A few warnings are expected. The test coverage report can be found at ```testcov_report.md``` in the main folder.

### Code organization
The source code implementation is contained in ```src``` and is organized as follows:

* ```src/activations.py``` implements bilipschitz activations and relative functionalities.
* ```src/blocks.py``` contains the implementation of some classes needed to build/complement the neural network architectures' skeleton.
* ```src/metrics.py``` implements metrics for training and testing.
* ```src/modules.py``` implements AE, SAE, SBAE, and SOAE networks along with their initialization procedures.
* ```src/NestedPOD.py``` comprises the definition of the homonymous class, useful for the EYS initialization.
* ```src/training.py``` contains the training loop function and relative utilities.
* ```src/utils.py``` implements additional utilities (e.g., for reading and saving files).

The scripts to run are contained in ```scripts```, whereas ```notebooks``` comprises the jupyter notebooks.

### Instructions
1. Generate the datasets by executing the notebook ```notebooks/datagen.ipynb```; the saved data are then available in ```data```.
2. Run the files contained in the ```scripts``` folder with ```python <file>.py```, whose results will be available in the folder ```results``` (see ```results/ckpts``` for the checkpoints and ```results/analyses``` for the raw analysis files);
3. Execute the remaining jupyter notebooks to visualize the numerical results and generate the paper figures, then available in the folder ```results/figures```.

### Cite
If the present repository and/or the original paper was useful in your research,
please consider citing
```
@misc{brivio2025saeeys,
      title={Deep Symmetric Autoencoders from the Eckart-Young-Schmidt Perspective},
      author={Simone Brivio and Nicola Rares Franco},
      year={2025},
      eprint={2506.11641},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2506.11641},
}
```
