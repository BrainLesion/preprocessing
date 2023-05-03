# BrainLes: A brain imaging toolkit beyond BraTS and lesion segmentation.

# niftyreg_preprocessor
does the following
* co-registration
* atlas registration
* transformation
* skullstripping in brats-space
* apply masking
* result: 4niftis, skullstripped brats-space -> save to path

## atlas
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/


## installation:
```
pip install -e "git+https://github.com/neuronflow/BrainLes.git@main#egg=HD_BET&subdirectory=preprocessing/brainles_preprocessing/brain_extraction/HD_BET"
```
