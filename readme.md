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
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Reference Final Resnet Model.ipynb"
   ```
3. Run the notebook to:
   - Initialize the WandB sweep
   - Train the ResNet model with different hyperparameter configurations
   - Log experiment results to WandB

## Hyperparameter Tuning
The model uses **random search** to optimize:
- Learning Rate (`0.001`, `0.0005`, `0.0001`)
- Batch Size (`32`, `64`, `128`)
- Optimizer (`adam`, `adadelta`, `rmsprop`)
- Weight Decay (`0.0`, `0.0001`, `0.001`)
- Fixed Epochs (`50`)

### Running a Sweep
Sweeps allow for automated hyperparameter tuning:
```python
sweep_id = wandb.sweep(sweep_config, project="a100-resnet-hyperparameter-tuning")
```

## Results
Training logs, loss curves, and accuracy metrics can be visualized in **WandB**.

### Confusion Matrix
![Confusion Matrix](WhatsApp%20Image%202025-03-14%20at%2015.09.26_6c2804ab.jpg)

### Training Accuracy
![Training Accuracy](WhatsApp%20Image%202025-03-13%20at%2017.33.05_fb425d9b.jpg)

### Training Loss
![Training Loss](WhatsApp%20Image%202025-03-13%20at%2017.33.05_9fe862a2.jpg)

### Hyperparameter Importance
![Hyperparameter Importance](WhatsApp%20Image%202025-03-13%20at%2017.35.54_beea379c.jpg)

## License
This project is licensed under the MIT License.

---

### Authors
Kirit Govindaraja Pillai - kx2222@nyu.edu  
Ruochong Wang - rw3760@nyu.edu  
Saketh Raman Ramesh - sr7714@nyu.edu  

---