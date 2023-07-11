from brainles_aurora.lib import single_inference

single_inference(
    t1c_file="Examples/BraTS-MET-00110-000-t1c.nii.gz",
    segmentation_file="your_segmentation_file.nii.gz",
    tta=False,  # optional: whether to use test time augmentations
    verbosity=True,  # optional: verbosity of the output
)
