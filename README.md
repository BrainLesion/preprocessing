

# BrainLes-Preprocessing
[![Python Versions](https://img.shields.io/pypi/pyversions/brainles-preprocessing)](https://pypi.org/project/brainles-preprocessing/)
[![Stable Version](https://img.shields.io/pypi/v/brainles-preprocessing?label=stable)](https://pypi.python.org/pypi/brainles-preprocessing/)
[![Documentation Status](https://readthedocs.org/projects/brainles-preprocessing/badge/?version=latest)](http://brainles-preprocessing.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/preprocessing/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- [![codecov](https://codecov.io/gh/BrainLesion/brainles-preprocessing/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/brainles-preprocessing) -->

`BrainLes preprocessing` is a comprehensive tool for preprocessing tasks in biomedical imaging, with a focus on (but not limited to) multi-modal brain MRI. It can be used to build modular preprocessing pipelines:

This includes **normalization**, **co-registration**, **atlas registration**, **skullstripping / brain extraction**, **N4 Bias correction** and **defacing**.
We provide means to transform images and segmentations in both directions between native and atlas space.

BrainLes is written modular and `backend-agnostic` meaning it allows to skip or swap registration, brain extraction, N4 bias correction and defacing tools.

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

# instantiate and run the preprocessor using defaults for backends (registration, brain extraction, bias correction, defacing)
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




## Citation

> [!IMPORTANT]
> If you use `brainles-preprocessing` in your research, please cite it to support the development!

Kofler, F., Rosier, M., Astaraki, M., Möller, H., Mekki, I. I., Buchner, J. A., Schmick, A., Pfiffer, A., Oswald, E., Zimmer, L., Rosa, E. de la, Pati, S., Canisius, J., Piffer, A., Baid, U., Valizadeh, M., Linardos, A., Peeken, J. C., Shit, S., … Menze, B. (2025). *BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis* [arXiv preprint arXiv:2507.09036](https://doi.org/10.48550/arXiv.2507.09036)


```
@misc{kofler2025brainlesionsuiteflexibleuserfriendly,
      title={BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis}, 
      author={Florian Kofler and Marcel Rosier and Mehdi Astaraki and Hendrik Möller and Ilhem Isra Mekki and Josef A. Buchner and Anton Schmick and Arianna Pfiffer and Eva Oswald and Lucas Zimmer and Ezequiel de la Rosa and Sarthak Pati and Julian Canisius and Arianna Piffer and Ujjwal Baid and Mahyar Valizadeh and Akis Linardos and Jan C. Peeken and Surprosanna Shit and Felix Steinbauer and Daniel Rueckert and Rolf Heckemann and Spyridon Bakas and Jan Kirschke and Constantin von See and Ivan Ezhov and Marie Piraud and Benedikt Wiestler and Bjoern Menze},
      year={2025},
      eprint={2507.09036},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.09036}, 
}
```

## Documentation
We provide a (WIP) documentation. Have a look [here](https://brainles-preprocessing.readthedocs.io/en/latest/?badge=latest)

## FAQ
Please credit the authors by citing their work.

### Registration
We currently fully support:
- [ANTs](https://github.com/ANTsX/ANTs) (default)
- [Niftyreg](https://github.com/KCL-BMEIS/niftyreg) (Linux)

We also offer basic support for:
- [greedy](https://greedy.readthedocs.io/en/latest/reference.html#greedy-usage) (Optional dependency, install via: `pip install brainles_preprocessing[picsl_greedy]`)
- [elastix](https://pypi.org/project/itk-elastix/0.13.0/) (Optional dependency, install via: `pip install brainles_preprocessing[itk-elastix]`)

As of now we do not offer inverse transforms for greedy and elastix. Please resort to ANTs or Niftyreg for this.

### Atlas Reference
We provide a range of different atlases via [zenodo](https://zenodo.org/records/15927391), namely:
-  [SRI24](https://www.nitrc.org/frs/download.php/4502/sri24_anatomy_unstripped_nifti.zip) and its [skull-stripped version](https://www.nitrc.org/frs/download.php/4499/sri24_anatomy_nifti.zip)
-  MNI152: [MNI_ICBM_2009c_Nonlinear_Symmetric](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/)
-  Slightly modified MNI152 and SRI24 atlas versions as employed for the [BraTS challenge algorithms](https://github.com/BrainLesion/BraTS)

> [!NOTE]  
> Custom atlases of your choice in NIfTI format are also supported


### N4 Bias correction
We currently provide support for N4 Bias correction based on [SimpleITK](https://simpleitk.org/)

### Brain extraction
We currently support:
- [HD-BET](https://github.com/MIC-DKFZ/HD-BET)
- [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (Optional dependency, install via: `pip install brainles_preprocessing[synthstrip]`)

### Defacing
We currently provide support for [Quickshear](https://github.com/nipy/quickshear).
