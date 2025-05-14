#!/bin/bash


# Read scene IDs from the Python list in scenes.txt
scene_ids=$(python3 -c "import ast; print(' '.join(str(i) for i in ast.literal_eval(open('scenes.txt').read())))")

#!/bin/bash

# DATASET_PATH="/Users/sahithyaravi/Documents/projectaria_tools/projectaria_tools_ase_data"
DATASET_PATH="/data/post_intern_sahithya/ariav5/projectaria_tools_ase_data/" 
OUTPUT_FOLDER="bird_eye_view"

# Loop through all scene folders (assumes each is named as a numeric scene ID)
for scene_dir in "$DATASET_PATH"/*/; do
    scene_id=$(basename "$scene_dir")
    echo "Running scene $scene_id..."
    python bev_per_scene.py --dataset_path "$DATASET_PATH" --scene_id "$scene_id" --output_folder "$OUTPUT_FOLDER"
    echo "Finished processing scene $scene_id."
done
