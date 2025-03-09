#!/bin/bash

# download the dataset from the source URL
# and save it to the data directory
# data/train
# data/validation

git clone https://github.com/parth1620/Facial-Expression-Dataset.git

CLONE_DIR="Facial-Expression-Dataset"
# Move data, clearing existing content
echo "Moving dataset to data/train and data/validation..."
mkdir -p data  # Ensure directories exist
rm -rf data/train data/validation  # Remove existing directories
mv "$CLONE_DIR/train" data/
mv "$CLONE_DIR/validation" data/

rm -rf "$CLONE_DIR"

echo "Dataset downloaded and moved to data/train and data/validation"
