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
for file in "$fixed" "$moving"; do
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist."
        exit 1
    fi
done

# Perform rigid-only registration with NiftyReg
niftyreg_path="brainles_preprocessing/registration/niftyreg_scripts/reg_aladin"
affine_options=("-rigOnly" \
    "-ref" "$fixed" \
    "-flo" "$moving" \
    "-res" "$transformed" \
    "-aff" "$matrix")

if [ -f "$niftyreg_path" ]; then
    "$niftyreg_path" "${affine_options[@]}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error: NiftyReg registration failed with exit code $exit_code."
        exit $exit_code
    fi
else
    echo "Error: NiftyReg script not found at '$niftyreg_path'."
    exit 1
fi

# Optional: Additional commands or post-processing steps

echo "Registration completed successfully."
