

# BrainLes-Preprocessing
[![Python Versions](https://img.shields.io/pypi/pyversions/brainles-preprocessing)](https://pypi.org/project/brainles-preprocessing/)
[![Stable Version](https://img.shields.io/pypi/v/brainles-preprocessing?label=stable)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/brainles-preprocessing/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/brainles-preprocessing) -->

`BrainLes preprocessing` is a comprehensive tool for preprocessing tasks in biomedical imaging, with a focus on (but not limited to) multi-modal brain MRI. It can be used to build modular preprocessing pipelines:

This includes **normalization**, **co-registration**, **atlas registration** and **skulstripping / brain extraction**.

BrainLes is written `backend-agnostic` meaning it allows to swap the registration, brain extraction tools and defacing tools.

<!-- TODO include image here -->


## Installation

With a Python 3.10+ environment you can install directly from [pypi.org](https://pypi.org/project/brainles-preprocessing/):

```
pip install brainles-preprocessing
```

We recommend using Python `3.10 / 3.11 / 3.12`.

> [!NOTE]  
> For python `3.13` the installation can currently fail with the error `Failed to build antspyx`.
> This usually means that there is no pre-built wheel for the package and it has to be build locally.
> This will require cmake (install e.g. with `pip install cmake`) and quite some time.
> Rerunning the installation with cmake installed should fix the error.


## Usage
A minimal example to register (to the standard atlas using ANTs) and skull strip (using HDBet) a t1c image (center modality) with 1 moving modality (flair) could look like this:
```python
from pathlib import Path
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import Preprocessor

patient_folder = Path("/home/marcelrosier/preprocessing/patient")

# specify a normalizer
percentile_normalizer = PercentileNormalizer(
    lower_percentile=0.1,
    upper_percentile=99.9,
    lower_limit=0,
    upper_limit=1,
)

# define center and moving modalities
center = CenterModality(
    modality_name="t1c",
    input_path=patient_folder / "t1c.nii.gz",
    normalizer=percentile_normalizer,
    # specify the output paths for the raw and normalized images of each step - here only for atlas registered and brain extraction
    raw_skull_output_path="patient/raw_skull_dir/t1c_skull_raw.nii.gz",
    raw_bet_output_path="patient/raw_bet_dir/t1c_bet_raw.nii.gz",
    raw_defaced_output_path="patient/raw_defaced_dir/t1c_defaced_raw.nii.gz",
    normalized_skull_output_path="patient/norm_skull_dir/t1c_skull_normalized.nii.gz",
    normalized_bet_output_path="patient/norm_bet_dir/t1c_bet_normalized.nii.gz",
    normalized_defaced_output_path="patient/norm_defaced_dir/t1c_defaced_normalized.nii.gz",
    # specify output paths for the brain extraction and defacing masks
    bet_mask_output_path="patient/masks/t1c_bet_mask.nii.gz",
    defacing_mask_output_path="patient/masks/t1c_defacing_mask.nii.gz",
)

moving_modalities = [
    Modality(
        modality_name="flair",
        input_path=patient_folder / "flair.nii.gz",
        normalizer=percentile_normalizer,
        # specify the output paths for the raw and normalized images of each step - here only for atlas registered and brain extraction
        raw_skull_output_path="patient/raw_skull_dir/fla_skull_raw.nii.gz",
        raw_bet_output_path="patient/raw_bet_dir/fla_bet_raw.nii.gz",
        raw_defaced_output_path="patient/raw_defaced_dir/fla_defaced_raw.nii.gz",
        normalized_skull_output_path="patient/norm_skull_dir/fla_skull_normalized.nii.gz",
        normalized_bet_output_path="patient/norm_bet_dir/fla_bet_normalized.nii.gz",
        normalized_defaced_output_path="patient/norm_defaced_dir/fla_defaced_normalized.nii.gz",
    )
]

# instantiate and run the preprocessor using defaults for registration/ brain extraction/ defacing backends
preprocessor = Preprocessor(
    center_modality=center,
    moving_modalities=moving_modalities,
)

preprocessor.run()

```


The package allows to choose registration backends, brain extraction tools and defacing methods.   
An example notebook with 4 modalities and further outputs and customizations can be found following these badges:

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

For further information please have a look at our [Jupyter Notebook tutorials](https://github.com/BrainLesion/tutorials/tree/main/preprocessing) in our tutorials repo (WIP).






<!-- TODO citation -->

## Documentation
We provide a (WIP) documentation. Have a look [here](https://brainles-preprocessing.readthedocs.io/en/latest/?badge=latest)

## FAQ
Please credit the authors by citing their work.

### Registration
We currently provide support for [ANTs](https://github.com/ANTsX/ANTs) (default), [Niftyreg](https://github.com/KCL-BMEIS/niftyreg) (Linux).

### Atlas Reference
We provide the SRI-24 atlas from this [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/).
However, custom atlases in NIfTI format are supported.

### Brain extraction
We currently provide support for [HD-BET](https://github.com/MIC-DKFZ/HD-BET).

### Defacing
We currently provide support for [Quickshear](https://github.com/nipy/quickshear).
