#!/bin/bash


# Read scene IDs from the Python list in scenes.txt
scene_ids=$(python3 -c "import ast; print(' '.join(str(i) for i in ast.literal_eval(open('scenes.txt').read())))")

#!/bin/bash

# DATASET_PATH="/Users/sahithyaravi/Documents/projectaria_tools/projectaria_tools_ase_data"
DATASET_PATH="/data/post_intern_sahithya/ariav5/projectaria_tools_ase_data/" 
OUTPUT_FOLDER="/data/post_intern_sahithya/top_down_views_questions"

# Loop through all scene folders (assumes each is named as a numeric scene ID)
for scene_id in $scene_ids; do

    echo "Running scene $scene_id..."
    python bev_per_scene.py --dataset_path "$DATASET_PATH" \
        --scene_id "$scene_id" \
        --output_folder "$OUTPUT_FOLDER" \
        --qa_file_path "/data/refactored-carnival/spatial_reasoning_qa_val_natural.json" 
    echo "Finished processing scene $scene_id."
done


REPO_DIR="/data/refactored-carnival"
cd $REPO_DIR

source ~/anaconda3/etc/profile.d/conda.sh
conda activate refactored-carnival


log="logs/run_gpt4o_3d_topdown.log"
out="output_3d/gpt4o/gpt4o_3d_topdown.csv"
bird_eye_folder="/data/post_intern_sahithya/top_down_views_questions"

python main_gpt4o.py \
    --file_path ./spatial_reasoning_qa_val_natural.json \
    --save_path "$out" \
    --use_bird_eye_scene_view \
    --bird_eye_path $bird_eye_folder \
    --mark_bbox \
    # --sample_size 600\
  2>&1 | tee logs/gpt4o_topdown.log