

# BrainLes-Preprocessing
[![Python Versions](https://img.shields.io/pypi/pyversions/brainles-preprocessing)](https://pypi.org/project/brainles-preprocessing/)
[![Stable Version](https://img.shields.io/pypi/v/brainles-preprocessing?label=stable)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/brainles-preprocessing/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/brainles-preprocessing) -->
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->

`BrainLes preprocessing` is a comprehensive tool for preprocessing tasks in biomedical imaging, with a focus on (but not limited to) multi-modal brain MRI. It can be used to build modular preprocessing pipelines:

This includes **normalization**, **co-registration**, **atlas registration** and **skulstripping / brain extraction**.

BrainLes is written `backend-agnostic` meaning it allows to swap the registration and brain extraction tools.

<!-- TODO mention defacing -->

<!-- TODO include image here -->


## Installation

With a Python 3.10+ environment you can install directly from [pypi.org](https://pypi.org/project/brainles-preprocessing/):

```
pip install brainles-preprocessing
```


## Usage
A minimal example can be found following these badges:  
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


For further information please have a look at our [Jupyter Notebook tutorials](https://github.com/BrainLesion/tutorials/tree/main/preprocessing) illustrating the usage of BrainLes preprocessing.


<!-- TODO citation -->

## Documentation
We provide a (WIP) documentation. Have a look [here](https://brainles-preprocessing.readthedocs.io/en/latest/?badge=latest)

## FAQ
Please credit the authors by citing their work.

### Atlas Reference
We provide the SRI-24 atlas from this [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/).
However, custom atlases can be supplied.

### Brain extraction
We currently provide support for [HD-BET](https://github.com/MIC-DKFZ/HD-BET).

### Registration
We currently provide support for [ANTs](https://github.com/ANTsX/ANTs) (default), [Niftyreg](https://github.com/KCL-BMEIS/niftyreg) (Linux), eReg (experimental)

<!-- TODO mention defacing -->
