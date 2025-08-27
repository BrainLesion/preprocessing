#!/bin/bash

# Function to check if a file exists
file_exists() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <niftyreg_executable> <fixed_image> <moving_image> <transformed_image> <transformation_matrix> <interpolation_method> <padding_value>"
    exit 1
fi

# Assign arguments to meaningful variable names
niftyreg_executable="$1"
fixed_image="$2"
moving_image="$3"
transformed_image="$4"
transformation_matrix="$5"
interpolation_method="$6"
padding_value="$7"

# Validate the existence of input files
if ! file_exists "$fixed_image"; then
    echo "Error: Fixed image '$fixed_image' does not exist."
    exit 1
fi

if ! file_exists "$moving_image"; then
    echo "Error: Moving image '$moving_image' does not exist."
    exit 1
fi

if ! file_exists "$transformation_matrix"; then
    echo "Error: Transformation matrix file '$transformation_matrix' does not exist."
    exit 1
fi

# NiftyReg configuration
niftyreg_path=$niftyreg_executable
# interpolation_method="3"  # Choose the appropriate interpolation method (e.g., 0, 1, 3)

# Perform resampling with NiftyReg
resample_command=(
    "$niftyreg_path"
    -ref "$fixed_image"
    -flo "$moving_image"
    -trans "$transformation_matrix"
    -res "$transformed_image"
    -inter "$interpolation_method"
    -pad "$padding_value"
)

if file_exists "$niftyreg_path"; then
    "${resample_command[@]}"
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
