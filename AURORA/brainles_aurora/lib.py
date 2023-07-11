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
    EnsureChannelFirstd,
)


def _turbo_path(the_path):
    turbo_path = Path(
        os.path.normpath(
            os.path.abspath(
                the_path,
            )
        )
    )
    return turbo_path


def _create_nifti_seg(
    threshold,
    reference_file,
    onehot_model_outputs_CHWD,
    output_file,
    whole_network_output_file,
    enhancing_network_output_file,
):

    # generate segmentation nifti
    activated_outputs = (
        (onehot_model_outputs_CHWD[0][:, :, :, :].sigmoid()).detach().cpu().numpy()
    )

    binarized_outputs = activated_outputs >= threshold

    binarized_outputs = binarized_outputs.astype(np.uint8)

    whole_metastasis = binarized_outputs[0]
    enhancing_metastasis = binarized_outputs[1]

    final_seg = whole_metastasis.copy()
    final_seg[whole_metastasis == 1] = 1  # edema
    final_seg[enhancing_metastasis == 1] = 2  # enhancing

    # get header and affine from T1
    REF = nib.load(reference_file)

    segmentation_image = nib.Nifti1Image(final_seg, REF.affine, REF.header)
    nib.save(segmentation_image, output_file)

    if whole_network_output_file:
        whole_network_output_file = Path(os.path.abspath(whole_network_output_file))

        whole_out = binarized_outputs[0]

        whole_out_image = nib.Nifti1Image(whole_out, REF.affine, REF.header)
        nib.save(whole_out_image, whole_network_output_file)

    if enhancing_network_output_file:
        enhancing_network_output_file = Path(
            os.path.abspath(enhancing_network_output_file)
        )

        enhancing_out = binarized_outputs[1]

        enhancing_out_image = nib.Nifti1Image(enhancing_out, REF.affine, REF.header)
        nib.save(enhancing_out_image, enhancing_network_output_file)


def _get_mode(
    t1_file,
    t1c_file,
    t2_file,
    fla_file,
):
    # t1
    if t1_file == None:
        t1_presence = False
    else:
        t1_presence = os.path.exists(t1_file)

    # t1c
    if t1c_file == None:
        t1c_presence = False
    else:
        t1c_presence = os.path.exists(t1c_file)

    # t2
    if t2_file == None:
        t2_presence = False
    else:
        t2_presence = os.path.exists(t2_file)

    # fla
    if fla_file == None:
        fla_presence = False
    else:
        fla_presence = os.path.exists(fla_file)

    print(
        f"t1: {t1_presence} t1c: {t1c_presence} t2: {t2_presence} flair: {fla_presence}"
    )

    if t1_presence and t1c_presence and t2_presence and fla_presence:
        mode = "t1-t1c-t2-fla"
    elif t1_presence and t1c_presence and fla_presence and not t2_presence:
        mode = "t1c-t1-fla"
    elif t1_presence and t1c_presence and not fla_presence and not t2_presence:
        mode = "t1c-t1"
    elif not t1_presence and t1c_presence and fla_presence and not t2_presence:
        mode = "t1c-fla"
    elif not t1_presence and t1c_presence and not fla_presence and not t2_presence:
        mode = "t1c-o"
    elif not t1_presence and not t1c_presence and fla_presence and not t2_presence:
        mode = "fla-o"
    elif t1_presence and not t1c_presence and not t2_presence and not fla_presence:
        mode = "t1-o"
    else:
        raise NotImplementedError("no model implemented for this combination of files")

    print("mode:", mode)
    return mode


def _get_dloader(
    mode,
    t1_file,
    t1c_file,
    t2_file,
    fla_file,
    workers,
):
    # T R A N S F O R M S
    if mode == "t1c-o" or "fla-o" or "t1-o":
        inference_transforms = Compose(
            [
                LoadImageD(keys=["images"]),
                EnsureChannelFirstd(keys="images"),
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
    else:
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

    if mode == "t1-t1c-t2-fla":
        images = [t1_file, t1c_file, t2_file, fla_file]

        the_dict = {
            "t1": t1_file,
            "t1c": t1c_file,
            "t2": t2_file,
            "fla": fla_file,
            "images": images,
        }

    elif mode == "t1c-t1-fla":
        images = [t1_file, t1c_file, fla_file]

        the_dict = {
            "t1": t1_file,
            "t1c": t1c_file,
            "fla": fla_file,
            "images": images,
        }

    elif mode == "t1c-t1":
        images = [t1_file, t1c_file]

        the_dict = {
            "t1": t1_file,
            "t1c": t1c_file,
            "images": images,
        }

    elif mode == "t1c-fla":
        images = [t1c_file, fla_file]

        the_dict = {
            "t1c": t1c_file,
            "fla": fla_file,
            "images": images,
        }

    elif mode == "t1c-o":
        images = [t1c_file]

        the_dict = {
            "t1c": t1c_file,
            "images": images,
        }

    elif mode == "fla-o":
        images = [fla_file]

        the_dict = {
            "fla": fla_file,
            "images": images,
        }

    elif mode == "t1-o":
        images = [t1_file]

        the_dict = {
            "t1": t1_file,
            "images": images,
        }
    else:
        raise NotImplementedError("no model implemented for this combination of files")

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

    return data_loader


def _get_model_and_weights(mode, model_selection):
    if mode == "t1-t1c-t2-fla":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )
        if model_selection == "best":
            weights = _turbo_path("model_weights/t1-t1c-t1-fla/t1-t1c-t1-fla_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1-t1c-t1-fla/t1-t1c-t1-fla_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "t1c-t1-fla":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/t1c-t1-fla/t1c-t1-fla_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1c-t1-fla/t1c-t1-fla_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "t1c-t1":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/t1c-t1/t1c-t1_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1c-t1/t1c-t1_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "t1c-fla":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/t1c-fla/t1c-fla_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1c-fla/t1c-fla_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "t1c-o":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/t1c-o/t1c-o_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1c-o/t1c-o_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "t1-o":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/t1-o/t1-o_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/t1-o/t1-o_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )

    elif mode == "fla-o":
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.1,
            act="mish",
        )

        if model_selection == "best":
            weights = _turbo_path("model_weights/fla-o/fla-o_best.tar")
        elif model_selection == "last":
            weights = _turbo_path("model_weights/fla-o/fla-o_last.tar")
        else:
            raise NotImplementedError(
                "no checkpoint implemented for this selection strategy."
            )
    else:
        raise NotImplementedError("no model implemented for this combination of files")

    return model, weights


# GO
def single_inference(
    segmentation_file,
    t1_file=None,
    t1c_file=None,
    t2_file=None,
    fla_file=None,
    whole_network_outputs_file=None,
    metastasis_network_outputs_file=None,
    cuda_devices="0",
    tta=True,
    sliding_window_batch_size=1, # faster for single interference (on RTX 3090)
    workers=0,
    threshold=0.5,
    sliding_window_overlap=0.5,
    crop_size=(192, 192, 32),
    model_selection="best",
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
    # ~~<< I N P U T S >>~~
    if t1_file is not None:
        t1_file = _turbo_path(t1_file)

    if t1c_file is not None:
        t1c_file = _turbo_path(t1c_file)

    if t2_file is not None:
        t2_file = _turbo_path(t2_file)

    if fla_file is not None:
        fla_file = _turbo_path(fla_file)

    # ~~<< M O D E >>~~
    mode = _get_mode(
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        fla_file=fla_file,
    )

    # ~~<< S E T T I N G S >>~~
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    multi_gpu = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory
    torch.cuda.empty_cache()

    data_loader = _get_dloader(
        mode=mode,
        t1_file=t1_file,
        t1c_file=t1c_file,
        t2_file=t2_file,
        fla_file=fla_file,
        workers=workers,
    )

    # ~~<< M O D E L >>~~
    model, model_weights = _get_model_and_weights(
        mode=mode,
        model_selection=model_selection,
    )
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

            try:
                reference_file = data["t1c"][0]
            except:
                try:
                    reference_file = data["fla"][0]
                except:
                    reference_file = data["t1"][0]
                else:
                    FileNotFoundError("no reference file found!")

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
