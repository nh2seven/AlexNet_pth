# AlexNet PyTorch Implementation
- PyTorch implementation of AlexNet for few-shot learning on the miniImageNet dataset. 
- This implementation uses the original AlexNet architecture with modifications for model-agnostic meta learning.

### Features
- PyTorch implementation of AlexNet
- Model-agnostic meta learning framework
- Few-shot learning support
- Training on [miniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) dataset
- Checkpoint management for model states
- Inference pipeline for predictions

### Requirements
- Python 3.11+
- PyTorch
- Kaggle API credentials

### Setup
Run setup.sh to fetch and organize the miniImageNet dataset:
```bash
chmod +x setup.sh
./setup.sh
```

### References
1. [AlexNet paper from NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
2. [Pytorch implementation of AlexNet by dansuh17](https://github.com/dansuh17/alexnet-pytorch)
3. [miniImageNet filelists](https://github.com/ashok-arjun/MLRC-2021-Few-Shot-Learning-And-Self-Supervision/tree/master/filelists/miniImagenet)