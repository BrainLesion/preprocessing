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
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <niftyreg_executable> <fixed_image> <moving_image> <transformed_image> <transformation_matrix> <padding_value>"
    exit 1
fi

# Assign arguments to meaningful variable names
niftyreg_executable="$1"
fixed_image="$2"
moving_image="$3"
transformed_image="$4"
transformation_matrix="$5"
padding_value="$6"

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
niftyreg_path=$niftyreg_executable
registration_options=(
    "-rigOnly"     # Perform rigid-only registration
    "-ref" "$fixed_image"
    "-flo" "$moving_image"
    "-res" "$transformed_image"
    "-aff" "$transformation_matrix"
    "-pad" "$padding_value"
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
