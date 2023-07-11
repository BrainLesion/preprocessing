from lib import single_inference
import click

@click.command()
@click.option('--t1_file', default=None, help='Path to T1 file')
@click.option('--t1c_file', default=None, help='Path to T1-CE file')
@click.option('--t2_file', default=None, help='Path to T2 file')
@click.option('--fla_file', default=None, help='Path to T2-FLAIR file')
@click.option('--segmentation_file', default=None, help='Path to segmentation output')
@click.option('--whole_network_outputs_file', default=None, help='OPTIONAL: Path to additional output for whole lesion')
@click.option('--metastasis_network_outputs_file', default=None, help='OPTIONAL: Path to additional output for metastasis only')
@click.option('--tta', default=True, help='Activate TTA')

def run_single_inference(
    t1_file=None,
    t1c_file=None,
    t2_file=None,
    fla_file=None,
    segmentation_file=None,
    whole_network_outputs_file=None,  # optional: whether to save network outputs for the whole metastasis (metastasis + edema)
    metastasis_network_outputs_file=None,  # optional: whether to save network outputs for the metastasis
    cuda_devices="0",  # optional: which CUDA devices to use
    tta=True,  # optional: whether to use test time augmentations
    sliding_window_batch_size=1,  # optional: adjust to fit your GPU memory, each step requires an additional 2 GB of VRAM
    workers=0,  # optional: workers for the data laoder
    threshold=0.5,  # optional: where to threshold the network outputs
    sliding_window_overlap=0.5,  # optional: overlap for the sliding window
    crop_size=(192, 192, 32),  # optional: only change if you know what you are doing
    model_selection="best",  # optional: only change if you know what you are doing
    verbosity=True,  # optional: verbosity of the output
):
    """Runs a single inference using all supplied sequences"""
    single_inference(
        t1_file,
        t1c_file,
        t2_file,
        fla_file,
        segmentation_file,
        whole_network_outputs_file,  # optional: whether to save network outputs for the whole metastasis (metastasis + edema)
        metastasis_network_outputs_file,  # optional: whether to save network outputs for the metastasis
        cuda_devices,  # optional: which CUDA devices to use
        tta,  # optional: whether to use test time augmentations
        sliding_window_batch_size,  # optional: adjust to fit your GPU memory, each step requires an additional 2 GB of VRAM
        workers,  # optional: workers for the data laoder
        threshold,  # optional: where to threshold the network outputs
        sliding_window_overlap,  # optional: overlap for the sliding window
        crop_size,  # optional: only change if you know what you are doing
        model_selection,  # optional: only change if you know what you are doing
        verbosity,  # optional: verbosity of the output
        )

if __name__ == "__main__":
    run_single_inference()

