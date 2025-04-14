# AlexNet_pth
A PyTorch implementation of AlexNet for few-shot classification using Model-Agnostic Meta-Learning (MAML).  
The model is trained on the [miniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) dataset using class-balanced episodic tasks.

## Features
- PyTorch implementation of AlexNet, adapted for MAML
- Meta-learning framework for few-shot tasks
- Support for N-way K-shot training episodes 
- Few-shot learning support
- Training on [miniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) dataset
- Checkpoint management for model states
- Inference pipeline for predictions

## Requirements
- Python 3.11+
- PyTorch 2.5+ (with CUDA 12.6)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials

## Setup
1. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate

    # OR

    conda create --name myenv python=3.11
    conda activate myenv
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and organize the dataset:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

## Usage
Run the meta-learning training process:
```bash
python main.py
```
You can modify training/evaluation settings in `config.yaml`.

## References
1. [AlexNet paper from NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
2. [Pytorch implementation of AlexNet by dansuh17](https://github.com/dansuh17/alexnet-pytorch)
3. [miniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)
4. [miniImageNet filelists](https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/tree/master/filelists/miniImagenet)
