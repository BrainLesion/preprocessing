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
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <fixed_image> <moving_image> <transformed_image> <transformation_matrix>"
    exit 1
fi

# Assign arguments to meaningful variable names
fixed_image="$1"
moving_image="$2"
transformed_image="$3"
transformation_matrix="$4"

# Validate the existence of input files
if ! file_exists "$fixed_image"; then
    echo "Error: Fixed image '$fixed_image' does not exist."
    exit 1
fi

if ! file_exists "$moving_image"; then
    echo "Error: Moving image '$moving_image' does not exist."
    exit 1
fi

# NiftyReg configuration
niftyreg_path="brainles_preprocessing/registration/niftyreg_scripts/reg_aladin"
registration_options=(
    "-rigOnly"     # Perform rigid-only registration
    "-ref" "$fixed_image"
    "-flo" "$moving_image"
    "-res" "$transformed_image"
    "-aff" "$transformation_matrix"
)

# Perform rigid-only registration with NiftyReg
if file_exists "$niftyreg_path"; then
    "$niftyreg_path" "${registration_options[@]}"
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
