# Deep Learning from Scratch: Step-by-Step Guide

Welcome to the **Deep Learning Python from Scratch** project. This repository demonstrates the implementation of a deep learning model **step by step** using only Python and fundamental libraries like `numpy`—without relying on any high-level deep learning frameworks such as TensorFlow or PyTorch.

---

## 🧠 Key Features

1. **Step-by-Step Implementation**  
   - The notebook breaks down each stage of building a neural network, from initializing parameters to optimizing the loss function.
   - Each step is explained: equations, and visualizations for clarity.

2. **Built from Scratch**  
   - No high-level libraries like TensorFlow or PyTorch are used. Instead, all computations are implemented manually using `numpy`.
   - This ensures a deep understanding of how deep learning works at low level.

3. **Educational Focus**  
   - Ideal for beginners who want to learn the fundamentals.
   - Great for developers looking to strengthen their understanding of deep learning theory and implementation.

---

## 🗂️ Project Structure
```
├── Datasets/                   # Data: divided in train, test, validation
├── explanations_utils/         # Resources for explaining in Jupyter (Ignore)
├── parameters_saved/           # Parameters of the model, once trained can be saved
├── utils/                      # Python code functions (auxiliary)
├── optimization_code.ipynb     # Methods to optimize main model
├── project.ipynb               # Main project and model
```

---

## 📊 Dataset and Objective

- Dataset Sources:

    - The training data for this project is sourced from:
        - The cats_vs_dogs catalog of TensorFlow (for cat images).
        - A dataset from Kaggle (for non-cat images).

- Objective:

    - This project is focused on implementing and understanding the fundamental workings of a neural network, rather than achieving state-of-the-art results.

    - The dataset and neural network are intentionally kept small and simple to prioritize educational value over accuracy or performance, Also, we do not make use of parallelism. (poor performance in training speed).