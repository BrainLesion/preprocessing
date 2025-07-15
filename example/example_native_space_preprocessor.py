from pathlib import Path

from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import NativeSpacePreprocessor


def preprocess(input_dir: Path, output_dir: Path):

    t1_file = list(input_dir.glob("*t1.nii.gz"))[0]
    t1c_file = list(input_dir.glob("*t1c.nii.gz"))[0]
    t2_file = list(input_dir.glob("*t2.nii.gz"))[0]
    flair_file = list(input_dir.glob("*fla.nii.gz"))[0]

    # Create output directories
    raw_bet_dir = output_dir / "raw_bet"
    raw_bet_dir.mkdir(parents=True, exist_ok=True)
    norm_bet_dir = output_dir / "normalized_bet"
    norm_bet_dir.mkdir(parents=True, exist_ok=True)
    raw_skull_dir = output_dir / "raw_skull"
    raw_skull_dir.mkdir(parents=True, exist_ok=True)
    norm_skull_dir = output_dir / "normalized_skull"
    norm_skull_dir.mkdir(parents=True, exist_ok=True)
    raw_deface_dir = output_dir / "raw_defaced"
    raw_deface_dir.mkdir(parents=True, exist_ok=True)

    percentile_normalizer = PercentileNormalizer()

    center = CenterModality(
        modality_name="t1c",
        input_path=t1c_file,
        raw_bet_output_path=raw_bet_dir / f"t1c_bet_raw.nii.gz",
        raw_skull_output_path=raw_skull_dir / f"t1c_skull_raw.nii.gz",
        normalized_skull_output_path=norm_skull_dir / f"t1c_skull_normalized.nii.gz",
        raw_defaced_output_path=raw_deface_dir / f"t1c_defaced_raw.nii.gz",
        normalizer=percentile_normalizer,
    )
    moving_modalities = [
        Modality(
            modality_name="t1",
            input_path=t1_file,
            raw_bet_output_path=raw_bet_dir / f"t1_bet_raw.nii.gz",
            raw_skull_output_path=raw_skull_dir / f"t1_skull_raw.nii.gz",
            normalized_skull_output_path=norm_skull_dir / f"t1_skull_normalized.nii.gz",
            raw_defaced_output_path=raw_deface_dir / f"t1_defaced_raw.nii.gz",
            normalizer=percentile_normalizer,
        ),
        Modality(
            modality_name="t2",
            input_path=t2_file,
            raw_bet_output_path=raw_bet_dir / f"t2_bet_raw.nii.gz",
            raw_skull_output_path=raw_skull_dir / f"t2_skull_raw.nii.gz",
            normalized_skull_output_path=norm_skull_dir / f"t2_skull_normalized.nii.gz",
            normalizer=percentile_normalizer,
        ),
        Modality(
            modality_name="flair",
            input_path=flair_file,
            raw_bet_output_path=raw_bet_dir / f"fla_bet_raw.nii.gz",
            raw_skull_output_path=raw_skull_dir / f"fla_skull_raw.nii.gz",
            normalized_skull_output_path=norm_skull_dir
            / f"fla_skull_normalized.nii.gz",
            raw_defaced_output_path=raw_deface_dir / f"fla_defaced_raw.nii.gz",
            normalizer=percentile_normalizer,
        ),
    ]

    preprocessor = NativeSpacePreprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
        limit_cuda_visible_devices="0",
    )

    preprocessor.run(
        save_dir_coregistration=output_dir / "coregistration",
        save_dir_n4_bias_correction=output_dir / "n4_bias_correction",
        save_dir_brain_extraction=output_dir / "brain_extraction",
        save_dir_defacing=output_dir / "defacing",
    )


if __name__ == "__main__":

    subject = "TCGA-DU-7294"  # "OtherEXampleFromTCIA"
    preprocess(
        input_dir=Path(
            f"/home/marcelrosier/preprocessing/example/example_data/{subject}"
        ),
        output_dir=Path(
            f"/home/marcelrosier/preprocessing/example/example_data/native_space_preprocessed_{subject}"
        ),
    )
