# BrainLes-Preprocessing

`BrainLes-Preprocessing` is a comprehensive tool tailored for preprocessing tasks in medical imaging, with a current focus on brain MRIs. Here's what it can currently do:

- **Co-registration using NiftyReg**: Aligning two images or series of images. While `NiftyReg` is the current tool used for co-registration, our architecture allows for potential extensions with other tools in the future.
- **Atlas Registration**: Maps images to a standard atlas for consistent spatial referencing.
- **Transformation**: Adjusts the image based on certain parameters.
- **Skull-stripping in BRATS-space**: Removes non-brain tissue from MRI data.
- **Apply Masking**: Applies a mask to an image, highlighting or hiding specific parts of it.

The outcome of this processing sequence is a set of 4 NIFTI images, skull-stripped in BRATS-space. These results are then saved to the provided path.

## Atlas Reference

The atlas employed in our workflow is based on the following publication:
[Link to the Atlas Article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2915788/)


## Installation


1. **Directly from the GitHub Repository**:
   Using pip, you can directly install the preprocessing tool:
   
   ```bash
   pip install git+https://github.com/BrainLesion/preprocessing.git
   ```

2. **Clone and Install Locally**:
   For a local installation, you can clone the repository and then install it:
   
   ```bash
   git clone https://github.com/BrainLesion/preprocessing.git
   cd preprocessing
   pip install .
   ```

## Directory Reference

`/home/florian/flow/BrainLesion/BrainLes/preprocessing`
