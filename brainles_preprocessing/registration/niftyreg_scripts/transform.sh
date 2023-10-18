# "reg_resample -ref " + ordner + pat + "/t1.nii.gz -flo " + ordner + pat + "/f2.nii.gz -trans " + ordner + pat + "/atlasreg.txt -res " + ordner + pat + "/f2.nii.gz -pad 0 -inter 3"
# rigid only registration with niftyreg
echo "fixed image: $1"
echo "moving image: $2"
echo "transformed image: $3"
echo "transformation matrix: $4"

fixed="$1"
moving="$2"
transformed="$3"
matrix="$4"

# niftyreg_scripts/reg_resample \
# TODO this path will be an issue for packaging
brainles_preprocessing/registration/niftyreg_scripts/reg_resample \
    -ref $fixed \
    -flo $moving \
    -trans $matrix \
    -res $transformed \
    -inter 3
    # -pad 0
