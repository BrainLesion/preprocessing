# basics
import os
import numpy as np
import nibabel as nib
from path import Path
from tqdm import tqdm

# import shutil
import time

# dl
import torch
from torch.utils.data import DataLoader

import monai
from monai.networks.nets import BasicUNet
from monai.data import list_data_collate
from monai.inferers import SlidingWindowInferer

from monai.transforms import RandGaussianNoised
from monai.transforms import (
    Compose,
    LoadImageD,
    Lambdad,
    ToTensord,
    ScaleIntensityRangePercentilesd,
)

from lib import cr


# GO
def missing_modality_inference(
    segmentation_file,
    t1_file=None,
    t1c_file=None,
    t2_file=None,
    fla_file=None,
    whole_network_outputs_file=None,
    metastasis_network_outputs_file=None,
    cuda_devices="0",
    tta=True,
    sliding_window_batch_size=20,
    workers=0,
    threshold=0.5,
    sliding_window_overlap=0.5,
    crop_size=(192, 192, 32),
    model_weights="model_weights/last_weights.tar",
    verbosity=True,
):
    """
    call this function to run the sliding window inference.

    Parameters:
    niftis: list of nifti files to infer
    comment: string to comment
    model_weights: Path to the model weights
    tta: whether to run test time augmentations
    threshold: threshold for binarization of the network outputs. Greater than <theshold> equals foreground
    cuda_devices: which cuda devices should be used for the inference.
    crop_size: crop size for the inference
    workers: how many workers should the data loader use
    sw_batch_size: batch size for the sliding window inference
    overlap: overlap used in the sliding window inference

    see the above function definition for meaningful defaults.
    """
    # ~~<< S E T T I N G S >>~~
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    multi_gpu = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory
    torch.cuda.empty_cache()

    # T R A N S F O R M S
    inference_transforms = Compose(
        [
            LoadImageD(keys=["images"]),
            Lambdad(["images"], np.nan_to_num),
            ScaleIntensityRangePercentilesd(
                keys="images",
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
                channel_wise=True,
            ),
            ToTensord(keys=["images"]),
        ]
    )
    # D A T A L O A D E R
    dicts = list()

    images = [t1_file, t1c_file, t2_file, fla_file]

    the_dict = {
        "t1": t1_file,
        "t1c": t1c_file,
        "t2": t2_file,
        "fla": fla_file,
        "images": images,
    }

    dicts.append(the_dict)

    # datasets
    inf_ds = monai.data.Dataset(data=dicts, transform=inference_transforms)

    # dataloaders
    data_loader = DataLoader(
        inf_ds,
        batch_size=1,
        num_workers=workers,
        collate_fn=list_data_collate,
        shuffle=False,
    )

    # ~~<< M O D E L >>~~
    model = BasicUNet(
        dimensions=3,
        in_channels=4,
        out_channels=2,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        act="mish",
    )

    model_weights = Path(os.path.abspath(model_weights))
    checkpoint = torch.load(model_weights, map_location="cpu")

    # inferer
    patch_size = crop_size

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sliding_window_batch_size,
        sw_device=device,
        device=device,
        overlap=sliding_window_overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # send model to device // very important for optimizer to work on CUDA
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load
    model.load_state_dict(checkpoint["model_state"])

    # epoch stuff
    if verbosity == True:
        time_date = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("start:", time_date)

    # limit batch length?!
    batchLength = 0

    # eval
    with torch.no_grad():
        model.eval()
        # loop through batches
        for counter, data in enumerate(tqdm(data_loader, 0)):
            if batchLength != 0:
                if counter == batchLength:
                    break

            # get the inputs and labels
            # print(data)
            # inputs = data["images"].float()
            inputs = data["images"]

            outputs = inferer(inputs, model)

            # test time augmentations
            if tta == True:
                n = 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                        "images"
                    ]

                    output = inferer(_img, model)
                    outputs = outputs + output
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img, dims=dims), model)

                        output = torch.flip(flip_pred, dims=dims)
                        outputs = outputs + output
                        n = n + 1.0
                outputs = outputs / n

            if verbosity == True:
                print("inputs shape:", inputs.shape)
                print("outputs:", outputs.shape)
                print("data length:", len(data))
                print("outputs shape 0:", outputs.shape[0])

            # generate segmentation nifti
            onehot_model_output = outputs

            reference_file = data["t1"][0]

            _create_nifti_seg(
                threshold=threshold,
                reference_file=reference_file,
                onehot_model_outputs_CHWD=onehot_model_output,
                output_file=segmentation_file,
                whole_network_output_file=whole_network_outputs_file,
                enhancing_network_output_file=metastasis_network_outputs_file,
            )

            # print("the time:", time.strftime("%Y-%m-%d_%H-%M-%S"))

    if verbosity == True:
        print("end:", time.strftime("%Y-%m-%d_%H-%M-%S"))


if __name__ == "__main__":
    pass
