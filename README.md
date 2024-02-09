[![PyPI version preprocessing](https://badge.fury.io/py/brainles-preprocessing.svg)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)


# BrainLes-Preprocessing
`BrainLes preprocessing` is a comprehensive tool for preprocessing tasks in biomedical imaging, with a focus on (but not limited to) multi-modal brain MRI. It can be used to build to build modular preprocessing pipelines:

This includes **normalization**, **co-registration**, **atlas registration** and **skulstripping / brain extraction**.

BrainLes is written `backend-agnostic` meaning it allows to swap the registration and brain extration tools.

<!-- TODO mention defacing -->

<!-- TODO include image here -->


## Installation

With a Python 3.10+ environment you can install directly from [pypi.org](https://pypi.org/project/brainles-preprocessing/):

```
pip install brainles-preprocessing
```


## Usage
Please have a look at our [Jupyter tutorials](https://TODO) illustrating the usage of BrainLes preprocessing.


<!-- TODO citation -->


## FAQ
Please credit the authors by citing their work.

### Atlas Reference
We provide the SRI-24 atlas from this [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/).
However, custom atlases can be supplied.

### Brain extraction
We currently provide support for [HD-BET](https://github.com/MIC-DKFZ/HD-BET).

### Registration
We currently provide support for [niftyreg](https://github.com/KCL-BMEIS/niftyreg).

<!-- TODO mention defacing -->
