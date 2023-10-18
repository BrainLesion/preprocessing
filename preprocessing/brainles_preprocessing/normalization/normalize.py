from auxiliary.nifti.io import read_nifti, write_nifti
from brainles_preprocessing.core import Modality
# TODO this is not working yet

def normalize_modality(modality: Modality):
    if modality.normalizer is not None:
        img = read_nifti(modality.current)
        normalized = modality.normalizer.normalize(image=img)
        write_nifti(
            input_array=normalized,
            output_nifti_path=modality.current,
            reference_nifti_path=modality.current,
            create_parent_directory=True,
        )
        # TODO
    return normalized
