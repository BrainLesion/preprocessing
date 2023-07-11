from lib import single_inference

single_inference(
    t1_file="*_t1.nii.gz",
    t1c_file="*_t1c.nii.gz",
    t2_file="*_t2.nii.gz",
    fla_file="*_fla.nii.gz",
    segmentation_file="your_segmentation_file.nii.gz",
    whole_network_outputs_file="your_whole_metastasis_file.nii.gz",  # optional: whether to save network outputs for the whole metastasis (metastasis + edema)
    metastasis_network_outputs_file="your_enhancing_metastasis_file.nii.gz",  # optional: whether to save network outputs for the metastasis
    cuda_devices="0",  # optional: which CUDA devices to use
    tta=True,  # optional: whether to use test time augmentations
    sliding_window_batch_size=4,  # optional: adjust to fit your GPU memory, each step requires an additional 2 GB of VRAM
    workers=0,  # optional: workers for the data laoder
    threshold=0.5,  # optional: where to threshold the network outputs
    sliding_window_overlap=0.5,  # optional: overlap for the sliding window
    crop_size=(192, 192, 32),  # optional: only change if you know what you are doing
    model_selection="best",  # optional: only change if you know what you are doing
    verbosity=True,  # optional: verbosity of the output
)
