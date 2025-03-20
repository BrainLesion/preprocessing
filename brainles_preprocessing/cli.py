from typing import Optional
from pathlib import Path
import typer
from typing_extensions import Annotated
from importlib.metadata import version


from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import Preprocessor


def version_callback(value: bool):
    __version__ = version("brainles_preprocessing")
    if value:
        typer.echo(f"Preprocessor CLI v{__version__}")
        raise typer.Exit()


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False
)


@app.command()
def main(
    input_t1c: Annotated[
        str,
        typer.Option(
            "-t1c",
            "--input_t1c",
            help="The path to the T1c image",
        ),
    ],
    input_t1: Annotated[
        str,
        typer.Option(
            "-t1",
            "--input_t1",
            help="The path to the T1 image",
        ),
    ],
    input_t2: Annotated[
        str,
        typer.Option(
            "-t2",
            "--input_t2",
            help="The path to the T2 image",
        ),
    ],
    input_fla: Annotated[
        str,
        typer.Option(
            "-fl",
            "--input_fla",
            help="The path to the FLAIR image",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "-o",
            "--output_dir",
            help="The path to the output directory",
        ),
    ],
    input_atlas: Annotated[
        Optional[str],
        typer.Option(
            "-a",
            "--input_atlas",
            help="The path to the atlas image",
        ),
    ] = "SRI24 BraTS atlas",
    version: Annotated[
        Optional[bool],
        typer.Option(
            "-v",
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
):
    """
    Preprocess the input images according to the BraTS protocol.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        input_path=input_t1c,
        normalizer=percentile_normalizer,
        # specify the output paths for the raw and normalized images of each step - here only for atlas registered and brain extraction
        raw_skull_output_path=output_dir / "t1c_skull_raw.nii.gz",
        raw_bet_output_path=output_dir / "t1c_bet_raw.nii.gz",
        raw_defaced_output_path=output_dir / "t1c_defaced_raw.nii.gz",
        normalized_skull_output_path=output_dir / "t1c_skull_normalized.nii.gz",
        normalized_bet_output_path=output_dir / "t1c_bet_normalized.nii.gz",
        normalized_defaced_output_path=output_dir / "t1c_defaced_normalized.nii.gz",
        # specify output paths for the brain extraction and defacing masks
        bet_mask_output_path=output_dir / "t1c_bet_mask.nii.gz",
        defacing_mask_output_path=output_dir / "t1c_defacing_mask.nii.gz",
    )

    for modality in ["t1", "t2", "fla"]:
        moving_modalities = [
            Modality(
                modality_name=modality,
                input_path=eval(f"input_{modality}"),
                normalizer=percentile_normalizer,
                # specify the output paths for the raw and normalized images of each step - here only for atlas registered and brain extraction
                raw_skull_output_path=output_dir / f"{modality}_skull_raw.nii.gz",
                raw_bet_output_path=output_dir / f"{modality}_bet_raw.nii.gz",
                raw_defaced_output_path=output_dir / f"{modality}_defaced_raw.nii.gz",
                normalized_skull_output_path=output_dir
                / f"{modality}_skull_normalized.nii.gz",
                normalized_bet_output_path=output_dir
                / f"{modality}_bet_normalized.nii.gz",
                normalized_defaced_output_path=output_dir
                / f"{modality}_defaced_normalized.nii.gz",
            )
        ]

    # if the input atlas is the SRI24 BraTS atlas, set it to None, because it will be picked up through the package
    if input_atlas == "SRI24 BraTS atlas":
        input_atlas = None

    # instantiate and run the preprocessor using defaults for registration/ brain extraction/ defacing backends
    preprocessor = Preprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
        temp_folder=output_dir / "temp",
        input_atlas=input_atlas,
    )

    preprocessor.run()


if __name__ == "__main__":
    app()
