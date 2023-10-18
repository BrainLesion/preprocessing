# rigid only registration with niftyreg
echo "fixed image: $1"
echo "moving image: $2"
echo "transformed image: $3"
echo "transformation matrix: $4"

fixed="$1"
moving="$2"
transformed="$3"
matrix="$4"

# niftyreg_scripts/reg_aladin \
# TODO this path will be an issue for packaging
brainles_preprocessing/registration/niftyreg_scripts/reg_aladin \
    -rigOnly \
    -ref $fixed \
    -flo $moving \
    -res $transformed \
    -aff $matrix
    # -pad 0
