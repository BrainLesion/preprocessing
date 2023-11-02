from abc import ABC, abstractmethod
import os
from .modality import Modality

class Preprocessor(ABC):
    def __init__(self, center_modality: Modality, moving_modalities: list[Modality], registrator, brain_extractor, temp_folder, atlas_image_path):
        self.center_modality = center_modality
        self.moving_modalities = moving_modalities
        self.temp_folder = temp_folder
        self.atlas_image_path = atlas_image_path

        self.registrator = registrator
        self.brain_extractor = brain_extractor

        self.coregistration_dir = os.path.join(self.temp_folder, "coregistration")
        os.makedirs(self.coregistration_dir, exist_ok=True)

        self.atlas_dir = os.path.join(self.temp_folder, "atlas-space")
        os.makedirs(self.atlas_dir, exist_ok=True)


    def run(self, brain_extraction:bool, normalization:bool):
        # Initializations: here just for now
        save_dir_coregistration = ...
        save_dir_atlas_registration = ...

        # Coregister moving modalities to center modality
        for moving_modality in self.moving_modalities:
            file_name = (
                f"/co__{self.center_modality.modality_name}__{moving_modality.modality_name}"
            )
            moving_modality.register(self.registrator, self.center_modality.current, self.coregistration_dir, file_name)
            
            if save_dir_coregistration:
                save_dir_coregistration = turbopath(save_dir_coregistration)
                native_cm = os.path.join(
                    self.coregistration_dir, f"native__{self.center_modality.modality_name}.nii.gz"
                )

                shutil.copyfile(self.center_modality.input_path, native_cm)
                shutil.copytree(self.coregistration_dir, save_dir_coregistration, dirs_exist_ok=True)
 
        # Register center modality to atlas
        file_name = f"atlas__{self.center_modality.modality_name}"
        self.center_modality.register(self.registrator, self.atlas_image_path, self.atlas_dir, file_name)

        # Transform moving modalities to atlas
        file_name = f"atlas__{moving_modality.modality_name}"
        for moving_modality in self.moving_modalities:
            moving_modality.transform(self.registrator, self.atlas_image_path, self.atlas_dir, file_name)

        # TODO: save steps in save_dir_atlas_registration

        # Optional: Brain extraction
        if brain_extraction:
            self.center_modality.extract_brain_region()
            for moving_modality in self.moving_modalities:
                if moving_modality.bet:
                    moving_modality.apply_mask()
        
        # Optional: Normalization
        if normalization:
            for modality in [self.center_modality] + self.moving_modalities:
                modality.normalize()

class PreprocessorGPU(Preprocessor):
    def __init__(self, limit_cuda_visible_devices: str = None):
        super().__init__()
        self.set_cuda_devices(limit_cuda_visible_devices)


    def set_cuda_devices(self, limit_cuda_visible_devices: str = None):
        if limit_cuda_visible_devices:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices



    