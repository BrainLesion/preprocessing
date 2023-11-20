import os
import shutil
import tempfile

from auxiliary.turbopath import turbopath

from .modality import Modality


class Preprocessor:
    def __init__(
        self,
        center_modality: Modality,
        moving_modalities: list[Modality],
        registrator,
        brain_extractor,
        temp_folder,
        atlas_image_path,
    ):
        self.center_modality = center_modality
        self.moving_modalities = moving_modalities
        self.atlas_image_path = atlas_image_path
        self.registrator = registrator
        self.brain_extractor = brain_extractor

        # create temporary storage
        if temp_folder:
            os.makedirs(temp_folder, exist_ok=True)
            self.temp_folder = turbopath(temp_folder)
        # custom temporary storage for debugging etc
        else:
            storage = tempfile.TemporaryDirectory()
            self.temp_folder = turbopath(storage.name)

        self.atlas_dir = os.path.join(self.temp_folder, "atlas-space")
        os.makedirs(self.atlas_dir, exist_ok=True)

    def run(
        self,
        brain_extraction: bool,
        normalization: bool,
        save_dir_coregistration: str = None,
        save_dir_atlas_registration: str = None,
        save_dir_brain_extraction: str = None,
        save_dir_unnormalized: str = None,
    ):
        # Coregister moving modalities to center modality
        coregistration_dir = os.path.join(self.temp_folder, "coregistration")
        os.makedirs(coregistration_dir, exist_ok=True)

        for moving_modality in self.moving_modalities:
            file_name = f"/co__{self.center_modality.modality_name}__{moving_modality.modality_name}"
            moving_modality.register(
                registrator=self.registrator,
                fixed_image_path=self.center_modality.current,
                registration_dir=coregistration_dir,
                moving_image_name=file_name,
            )
        self._save_coregistration(save_dir_coregistration)

        # Register center modality to atlas
        file_name = f"atlas__{self.center_modality.modality_name}"
        self.center_modality.register(
            registrator=self.registrator,
            fixed_image_path=self.atlas_image_path,
            registration_dir=self.atlas_dir,
            moving_image_name=file_name,
        )

        # Transform moving modalities to atlas
        for moving_modality in self.moving_modalities:
            file_name = f"atlas__{moving_modality.modality_name}"
            moving_modality.transform(
                registrator=self.registrator,
                fixed_image_path=self.atlas_image_path,
                registration_dir=self.atlas_dir,
                moving_image_name=file_name,
            )
        self._save_output(self.atlas_dir, save_dir_atlas_registration)

        # Optional: Brain extraction
        if brain_extraction:
            bet_dir = os.path.join(self.temp_folder, "brain-extraction")
            brain_masked_dir = os.path.join(bet_dir, "brain_masked")
            atlas_mask = self.center_modality.extract_brain_region(
                brain_extractor=self.brain_extractor,
                bet_dir=bet_dir,
                atlas_mask=atlas_mask,
            )
            for moving_modality in self.moving_modalities:
                moving_modality.apply_mask(
                    brain_extraction=self.brain_extractor,
                    brain_masked_dir=brain_masked_dir,
                    atlas_mask=atlas_mask,
                )

            self._save_output(bet_dir, save_dir_brain_extraction)

        # Optional: Normalization
        if normalization:
            for modality in [self.center_modality] + self.moving_modalities:
                modality.normalize(self.temp_folder, save_dir_unnormalized)

    def _save_output(self, src, save_dir):
        if save_dir:
            save_dir = turbopath(save_dir)
            shutil.copytree(src, save_dir, dirs_exist_ok=True)

    def _save_coregistration(self, coregistration_dir):
        if save_dir_coregistration:
            save_dir_coregistration = turbopath(save_dir_coregistration)
            native_cm = os.path.join(
                coregistration_dir,
                f"native__{self.center_modality.modality_name}.nii.gz",
            )

            shutil.copyfile(self.center_modality.input_path, native_cm)
            shutil.copytree(
                coregistration_dir, save_dir_coregistration, dirs_exist_ok=True
            )


class PreprocessorGPU(Preprocessor):
    def __init__(
        self,
        center_modality: Modality,
        moving_modalities: list[Modality],
        registrator,
        brain_extractor,
        temp_folder,
        atlas_image_path,
        limit_cuda_visible_devices: str = None,
    ):
        super().__init__(
            center_modality,
            moving_modalities,
            registrator,
            brain_extractor,
            temp_folder,
            atlas_image_path,
        )
        self.set_cuda_devices(limit_cuda_visible_devices)

    def set_cuda_devices(self, limit_cuda_visible_devices: str = None):
        if limit_cuda_visible_devices:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices
