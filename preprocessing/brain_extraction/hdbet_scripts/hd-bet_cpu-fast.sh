# cpu skullstripping with HD-bet
echo "input image: $1"
echo "masked image: $2"

input_image="$1"
masked_image="$2"

hd-bet \
    -i $input_image \
    -o $masked_image \
    -s 1 \
    -d cpu \
    -mode fast \
    -tta 0
