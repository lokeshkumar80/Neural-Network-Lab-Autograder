
# 🧩 Neural Network Lab – Autograder & Implementation

This repository implements a **Neural Network from scratch using NumPy**, covering forward and backward propagation, convolution, and pooling layers.  
It includes an **autograder** for automated testing and multiple experiments on different datasets.

---

## 🚀 Features

- Fully implemented from scratch (no TensorFlow/PyTorch)
- Feedforward and backpropagation for:
  - `FullyConnectedLayer`
  - `ConvolutionLayer`
  - `AvgPoolingLayer`
  - `FlattenLayer`
- Custom training loop, validation, and model saving/loading
- Autograding framework for automatic evaluation of all tasks
- Dataset utilities and visualization scripts
- Experiments on **Square**, **SemiCircle**, **MNIST**, and **CIFAR-10**

---

## 📊 Tasks Overview

| Task | Dataset | Description | Accuracy |
|------|----------|--------------|-----------|
| 2.1 | Square | Linear separation | ~98% |
| 2.2 | Semi-Circle | Non-linear boundary | ~98% |
| 2.3 | MNIST | Digit classification | ~91% |
| 2.4 | CIFAR-10 | Image classification (ConvNet) | ~35–40% |

> Full experimental analysis is included in [`observations.txt`](observations.txt).

---

## 🧩 File Descriptions

| File | Description |
|------|--------------|
| `autograder.py` | Runs automated grading and scoring for tasks |
| `layers.py` | Implements layer classes with forward/backward logic |
| `nn.py` | Defines `NeuralNetwork` class (training, validation, accuracy) |
| `tasks.py` | Defines model architectures for each dataset |
| `test.py` | Runs specified task interactively |
| `test_feedforward.py` | Verifies correctness of feedforward operations |
| `util.py` | Handles data reading, preprocessing, and one-hot encoding |
| `visualize.py` | Visualization utilities for 2D datasets |
| `visualizeTruth.py` | Visualizes dataset ground truth boundaries |
| `observations.txt` | Contains experiment results and observations |
| `reference.txt` | Modification and reference log |
| `NNL_Testing.ipynb` | Notebook for testing and debugging models |

---

## ⚙️ How to Run

### **Run Individual Tasks**
```bash
python3 test.py 1 42    # Runs Task 2.1 (Square)
python3 test.py 2 42    # Runs Task 2.2 (SemiCircle)
python3 test.py 3 42    # Runs Task 2.3 (MNIST)
python3 test.py 4 42    # Runs Task 2.4 (CIFAR-10)

python3 autograder.py -t 1    # Task 1: Forward Pass
python3 autograder.py -t 2    # Task 2: Full Backpropagation

python3 visualizeTruth.py 1   # Square Dataset
python3 visualizeTruth.py 2   # SemiCircle Dataset

```
### 📚 References

- Numpy Documentation: https://numpy.org/doc/

- CS231n: Convolutional Neural Networks (Stanford University)

- Deep Learning by Ian Goodfellow et al.

- ChatGPT (OpenAI, 2025) — guidance on debugging and architecture tuning
