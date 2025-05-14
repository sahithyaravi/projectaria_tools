#!/bin/bash

# Read scene IDs from the Python list in scenes.txt
scene_ids=$(python3 -c "import ast; print(' '.join(str(i) for i in ast.literal_eval(open('scenes.txt').read())))")

# Other config variables
SET="train"
CDN_FILE="aria_synthetic_environments_dataset_download_urls.json"
OUTPUT_DIR="projectaria_tools_ase_data"
UNZIP="True"

# Filter out scene IDs that already exist in the output directory
missing_ids=""
for id in $scene_ids; do
  if [ ! -d "$OUTPUT_DIR/$SET/scene_$id" ]; then
    missing_ids+="$id,"
  fi
done

# Remove trailing comma
missing_ids="${missing_ids%,}"

# Download only if there are missing IDs
if [ -n "$missing_ids" ]; then
  echo "Downloading missing scene IDs: $missing_ids"
  python3 projects/AriaSyntheticEnvironment/aria_synthetic_environments_downloader.py \
    --set $SET \
    --scene-ids $missing_ids \
    --cdn-file $CDN_FILE \
    --output-dir $OUTPUT_DIR \
    --unzip $UNZIP
else
  echo "All scene IDs are already downloaded."
fi