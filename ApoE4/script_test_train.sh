#!/bin/bash

# Load AmberTools module (adjust if needed)
module add amber-16

# SET VARIABLES
TOPOLOGY="filtered.pdb"  # Topology file
STARTRES=1
ENDRES=299
STARTOUTRES=24
ENDOUTRES=161
OFFSET=50
NO_OUTPUT_FRAMES=19  # Number of frames to output

# Create output directories
mkdir -p train test

# Detect and sort folders correctly
FOLDERS=($(ls -d e[0-9]*s[0-9]* 2>/dev/null | sort -V))

# Print found folders for debugging
echo "Detected Folders:"
printf '%s\n' "${FOLDERS[@]}"

# Check if folders were found
TOTAL_FOLDERS=${#FOLDERS[@]}
if [ "$TOTAL_FOLDERS" -eq 0 ]; then
    echo "ERROR: No valid folders found! Check folder names and location."
    exit 1
fi

TRAIN_SIZE=$((TOTAL_FOLDERS * 80 / 100))
TEST_SIZE=$((TOTAL_FOLDERS - TRAIN_SIZE))

echo "Total folders: $TOTAL_FOLDERS"
echo "Train folders: $TRAIN_SIZE"
echo "Test folders: $TEST_SIZE"

# Function to perform subsampling
subsample() {
    local folder=$1
    local output_dir=$2  # Either "train" or "test"
    local ignore_fraction=10  # Ignore first 10% of snapshots only for test set

    for xtc in $(ls "$folder"/*.xtc 2>/dev/null); do
        [ -f "$xtc" ] || continue  # Skip if no XTC files

        # Count frames
        sed "s|TRAJIN|$xtc|" tmpfolder/countframes.in > countframes.in
        cpptraj -p $TOPOLOGY -i countframes.in > countframes_output
        TOTAL_FRAMES=$(grep "will occur on [0-9]\+ frames." countframes_output | grep -o "[0-9]\+")
        
        if [ "$output_dir" == "test" ]; then
            IGNORE_FRAMES=$((TOTAL_FRAMES * ignore_fraction / 100))
            STARTFRAME=$((IGNORE_FRAMES + 1))  # Start after ignoring 10%
        else
            STARTFRAME=1  # Train set starts from frame 1
        fi

        ENDFRAME=$((STARTFRAME + NO_OUTPUT_FRAMES * OFFSET))
        mv countframes_output $output_dir/$(basename "$folder").$(basename "$xtc").countframes_output

        echo "$folder/$xtc will start at frame $STARTFRAME with offset $OFFSET"

        sed "s|PARM|$TOPOLOGY|g; s|TRAJIN|$xtc|g; s|OUTFOLDER|$output_dir|g; s|OUTPREFIX|output|g; s|FOLDERXTC|$(basename "$folder").$(basename "$xtc")|g; s|STARTRES|$STARTRES|g; s|ENDRES|$ENDRES|g; s|STARTOUTRES|$STARTOUTRES|g; s|ENDOUTRES|$ENDOUTRES|g; s|STARTFRAME|$STARTFRAME|g; s|OFFSET|$OFFSET|g; s|ENDFRAME|$ENDFRAME|g" tmpfolder/_cpptraj_strip_align.in > cpptraj.in.tmp
        
        cpptraj -p $TOPOLOGY -i cpptraj.in.tmp > $output_dir/output.$(basename "$folder").$(basename "$xtc").cpptraj.out
        mv cpptraj.in.tmp $output_dir/output.$(basename "$folder").$(basename "$xtc").cpptraj.in
    done
}

# Process the first 80% for training
for ((i = 0; i < TRAIN_SIZE; i++)); do
    subsample "${FOLDERS[$i]}" "train"
done

# Process the last 20% for testing, ignoring the first 10% of snapshots in each folder
for ((i = TRAIN_SIZE; i < TOTAL_FOLDERS; i++)); do
    subsample "${FOLDERS[$i]}" "test"
done

echo "Subsampling complete! Train and test data saved."