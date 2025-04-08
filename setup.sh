#!/bin/bash
set -e

# Create directories
mkdir -p data/filelists

# Download the miniImagenet dataset from Kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle/"
    exit 1
fi

# Download and extract dataset
(cd data && \
    kaggle datasets download arjunashok33/miniimagenet && \
    unzip miniimagenet.zip && \
    rm miniimagenet.zip)

# Download the filelists for miniImagenet
(cd data/filelists && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/base.json && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/val.json && \
    curl -O https://raw.githubusercontent.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/master/filelists/miniImagenet/novel.json)

echo "Setup completed successfully!"