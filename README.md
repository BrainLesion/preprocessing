[![PyPI version preprocessing](https://badge.fury.io/py/brainles-preprocessing.svg)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)


# BrainLes-Preprocessing

`BrainLes-Preprocessing` is a comprehensive tool tailored for preprocessing tasks in medical imaging, with a current focus on brain MRIs. Here's what it can currently do:

- **Co-registration using NiftyReg**: Aligning two images or series of images. While `NiftyReg` is the current tool used for co-registration, our architecture allows for potential extensions with other tools in the future.
- **Atlas Registration**: Maps images to a standard atlas for consistent spatial referencing.
- **Transformation**: Adjusts the image based on certain parameters.
- **Skull-stripping in BRATS-space**: Removes non-brain tissue from MRI data.
- **Apply Masking**: Applies a mask to an image, highlighting or hiding specific parts of it.

The outcome of this processing sequence is a set of 4 NIFTI images, skull-stripped in BRATS-space. These results are then saved to the provided path.

## Atlas Reference

We use the SRI-24 atlas from this [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/)

## Installation

With a Python 3.10+ environment you can install directly from [pypi.org](https://pypi.org/project/brainles-preprocessing/):

```
pip install brainles-preprocessing
```
