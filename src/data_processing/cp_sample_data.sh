#!/bin/bash

# Source directories
IMAGE_SOURCE_DIR="/run/media/snau/general/capstone/Post_images/image"
TEXT_SOURCE_DIR="/run/media/snau/general/capstone/Post_metadata/info"

# Destination directories
IMAGE_DEST_DIR="$HOME/ha/image"
INFO_DEST_DIR="$HOME/ha/info"
# IMAGE_DEST_DIR="/run/media/snau/04E884E1E884D1FA/image"
# INFO_DEST_DIR="/run/media/snau/04E884E1E884D1FA/info"

while IFS=$'\t' read -r influencer_name json_file image_files; do
    # Remove brackets and extra characters from filenames
    json_file=$(echo "$json_file" | tr -d '[]' | xargs)
    image_files=$(echo "$image_files" | tr -d "[]" | tr ',' ' ' | xargs)
   
    echo "Processing JSON: $json_file"

    # Copy JSON file
    if [[ -f "$TEXT_SOURCE_DIR/$influencer_name-$json_file" ]]; then
        cp "$TEXT_SOURCE_DIR/$influencer_name-$json_file" "$INFO_DEST_DIR/"
        echo "Copied JSON: $json_file"
    else
        echo "Warning: JSON file not found -> $TEXT_SOURCE_DIR/$influencer_name-$json_file"
    fi

    # Copy Image files
    for image in $image_files; do
        image=$(echo "$image" | xargs)  # Trim spaces
        echo "Processing Image: $image"
        if [[ -f "$IMAGE_SOURCE_DIR/$influencer_name-$image" ]]; then
            cp "$IMAGE_SOURCE_DIR/$influencer_name-$image" "$IMAGE_DEST_DIR/"
            echo "Copied Image: $image"
        else
            echo "Warning: Image file not found -> $IMAGE_SOURCE_DIR/$influencer_name-$image"
        fi

    done

done < small_mapping.txt

echo "File transfer completed!"
