# ResNet Model with Hyperparameter Tuning

This repository contains a Jupyter Notebook that trains a **ResNet model on the CIFAR-10 dataset** with hyperparameter tuning using **Weights & Biases (WandB)**. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The project focuses on optimizing training parameters through automated sweeps to achieve the best classification performance. By leveraging **PyTorch**, **WandB**, and **random search sweeps**, we fine-tune key hyperparameters such as learning rate, batch size, optimizer type, and weight decay to improve model accuracy and reduce training loss.

## Features
- Uses **WandB** for experiment tracking and hyperparameter tuning.
- Implements **ResNet** for image classification.
- Includes **random search sweeps** for optimizing learning rate, batch size, optimizer type, and weight decay.
- Applies **PyTorch transforms** for data preprocessing.

## Installation
To set up the environment, install the required dependencies:

```bash
pip install torch torchvision wandb
```

## Running the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/resnet-hyperparameter-tuning.git
   cd resnet-hyperparameter-tuning
   ```
2. Run the notebook to:
   - Initialize the WandB sweep
   - Train the ResNet model with different hyperparameter configurations
   - Log experiment results to WandB

## Hyperparameter Tuning
The model uses **random search** to optimize:
- Learning Rate (`0.001`, `0.0005`, `0.0001`)
- Batch Size (`32`, `64`, `128`)
- Optimizer (`adam`, `adadelta`, `rmsprop`)
- Weight Decay (`0.0`, `0.0001`, `0.001`)
- Fixed Epochs (`300`)

### Running a Sweep
Sweeps allow for automated hyperparameter tuning:
```python
wandb.sweep(sweep_config, project="a100-resnet-hyperparameter-tuning")
```

## Results
Accuracy metrics, loss curves, and hyperparameter relevance were visualized in **WandB**.

### Confusion Matrix
<img src="https://github.com/user-attachments/assets/429b6c42-217c-4415-aeaf-156398e691d8" width="600"/>

### Training Accuracy

<img src="https://github.com/user-attachments/assets/8c70988c-20f8-49ef-a128-0dd34becf7af" width="600"/>

### Training Loss

<img src="https://github.com/user-attachments/assets/026202b0-d556-486a-b678-c8d49ce9c37e" width="600"/>

### Hyperparameter Importance

<img src="https://github.com/user-attachments/assets/cedf485a-55c3-463e-bffd-25c2036f7c6c" width="600"/>

---

### Authors
Kirit Govindaraja Pillai - kx2222@nyu.edu  
Ruochong Wang - rw3760@nyu.edu  
Saketh Raman Ramesh - sr7714@nyu.edu  

---
