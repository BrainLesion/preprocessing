

# BrainLes-Preprocessing
[![Python Versions](https://img.shields.io/pypi/pyversions/brainles-preprocessing)](https://pypi.org/project/brainles-preprocessing/)
[![Stable Version](https://img.shields.io/pypi/v/brainles-preprocessing?label=stable)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/brainles-preprocessing/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/brainles-preprocessing) -->

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
A minimal example to register (to the standard atlas using ANTs) and skull strip (using HDBet) a t1c image (center modality) with 1 moving modality (flair) could look like this:
```python
from pathlib import Path
from auxiliary.normalization.percentile_normalizer import PercentileNormalizer

from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.preprocessor import Preprocessor
from brainles_preprocessing.registration import ANTsRegistrator  # , NiftyRegRegistrator,# eRegRegistrator

patient_folder = Path("/home/marcelrosier/preprocessing/patient")

# specify a normalizer
percentile_normalizer = PercentileNormalizer(
    lower_percentile=0.1,
    upper_percentile=99.9,
    lower_limit=0,
    upper_limit=1,
)

# define modalities
center = Modality(
    modality_name="t1c",
    input_path=patient_folder / "t1c.nii.gz",
    # specify the output paths for the raw and normalized images of each step (all optional)
    raw_bet_output_path="patient/raw_bet_dir/t1c_bet_raw.nii.gz",
    raw_skull_output_path="patient/raw_skull_dir/t1c_skull_raw.nii.gz",
    normalized_bet_output_path="patient/norm_bet_dir/t1c_bet_normalized.nii.gz",
    normalized_skull_output_path="patient/norm_skull_dir/t1c_skull_normalized.nii.gz",
    atlas_correction=True,
    normalizer=percentile_normalizer,
)

moving_modalities = [
    Modality(
        modality_name="flair",
        input_path=patient_folder / "flair.nii.gz",
        # specify the output paths for the raw and normalized images of each step (all optional)
        raw_bet_output_path="patient/raw_bet_dir/fla_bet_raw.nii.gz",
        raw_skull_output_path="patient/raw_skull_dir/fla_skull_raw.nii.gz",
        normalized_bet_output_path="patient/norm_bet_dir/fla_bet_normalized.nii.gz",
        normalized_skull_output_path="patient/norm_skull_dir/fla_skull_normalized.nii.gz",
        atlas_correction=True,
        normalizer=percentile_normalizer,
    )
]

preprocessor = Preprocessor(
    center_modality=center,
    moving_modalities=moving_modalities,
    # choose the registration backend & brain extractor you want to use
    registrator=ANTsRegistrator(),
    brain_extractor=HDBetExtractor(),
    limit_cuda_visible_devices="0",
)

preprocessor.run(
    save_dir_coregistration="output/co-registration",
    save_dir_atlas_registration="output/atlas-registration",
    save_dir_atlas_correction="output/atlas-correction",
    save_dir_brain_extraction="output/brain-extraction",
)
```


An example notebook with 4 modalities can be found following these badges:

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/BrainLesion/tutorials/blob/main/preprocessing/preprocessing_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

For further information please have a look at our [Jupyter Notebook tutorials](https://github.com/BrainLesion/tutorials/tree/main/preprocessing) in our tutorials repo.






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
