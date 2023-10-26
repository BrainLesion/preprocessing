#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <fixed_image> <moving_image> <transformed_image> <transformation_matrix>"
    exit 1
fi

# Assign arguments to variables
fixed="$1"
moving="$2"
transformed="$3"
matrix="$4"

# Check if input files exist
for file in "$fixed" "$moving" "$matrix"; do
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist."
        exit 1
    fi
done

# Perform resampling with NiftyReg
niftyreg_path="brainles_preprocessing/registration/niftyreg_scripts/reg_resample"
resample_options=(
    "-ref" "$fixed"
    "-flo" "$moving"
    "-trans" "$matrix"
    "-res" "$transformed"
    "-inter 3"
)

if [ -f "$niftyreg_path" ]; then
    "$niftyreg_path" "${resample_options[@]}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error: NiftyReg failed with exit code $exit_code."
        exit $exit_code
    fi
else
    echo "Error: NiftyReg script not found at '$niftyreg_path'."
    exit 1
fi

# Optional: Additional commands or post-processing steps

echo "Resampling completed successfully."
