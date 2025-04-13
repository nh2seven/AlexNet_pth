#!/bin/bash
set -e

# Create directories
mkdir -p data/filelists
mkdir -p data/filelists_clean
mkdir -p data/filelists_raw

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle/"
    exit 1
fi

# Download and extract miniImageNet dataset
if [ -z "$(ls -A data/miniImageNet)" ]; then
    echo "Downloading miniImageNet dataset from Kaggle into data/miniImageNet/..."
    (cd data/miniImageNet && \
        kaggle datasets download arjunashok33/miniimagenet && \
        unzip -q miniimagenet.zip && \
        rm miniimagenet.zip)
else
    echo "Dataset already exists in data/miniImageNet/, skipping download."
fi

# Download original JSON filelists
echo "Downloading original filelists..."
(cd data/filelists && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/base.json && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/val.json && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/novel.json)

# Backup + Clean JSON paths
echo "Cleaning JSON paths..."
for file in base.json val.json novel.json; do
    cp data/filelists/$file data/filelists_raw/$file
    jq '."image_names" |= map(sub("filelists/miniImagenet/"; ""))' data/filelists/$file > data/filelists_clean/$file
done

echo "Setup completed successfully."
