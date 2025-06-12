# Tahoe-100M CVAE for Drug Response Prediction

## Overview

This project contains code and Jupyter notebooks to train and evaluate a **Conditional Variational Autoencoder (CVAE)** on a subset of the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M). The goal is to model single-cell gene expression changes induced by drug perturbations, enabling better understanding of transcriptional response to treatment.

We explore the Tahoe-100M dataset (focusing on a 1M cell subset), preprocess the data, define and train a CVAE model, and analyze the results both interactively and via production-ready scripts.

---

## Project Structure

```
.
├── data/                # Raw and processed data
├── models/              # Saved trained models
├── notebooks/           # Jupyter notebooks for exploration
│   ├── loading_data.ipynb
│   └── model_dev.ipynb
├── src/                 # Source code for production usage
│   ├── data_utils.py    # Data loading and preprocessing
│   ├── model.py         # CVAE model architecture
│   ├── train.py         # Training script
│   └── predict.py       # Prediction script
├── requirements.txt     # Python dependencies
├── environment.yaml     # Conda environment
└── README.md
```

---

## Notebooks

The `notebooks/` folder contains two primary notebooks:

### 1. `loading_data.ipynb`
Purpose: Data preprocessing and preparation  
Main tasks:
- Import required libraries
- Load raw data from the Tahoe-100M dataset
- Clean and preprocess expression matrices and metadata
- Feature engineering and data validation
- Save processed data to disk for training

### 2. `model_dev.ipynb`
Purpose: Model development, training, and evaluation  
Main tasks:
- Load preprocessed data
- Define CVAE model architecture
- Implement training loop and evaluation metrics
- Visualize training loss and prediction accuracy
- Analyze predictions on a hold-out set
- Save best-performing models to `models/` directory

---

## Production Code

The `src/` directory contains modular Python scripts for reproducible and scalable training and inference.

### Training

To train the CVAE model from preprocessed data:
```bash
python src/train.py
```

### Prediction

To use a trained model for predictions:
```bash
python src/predict.py
```

---

## Setup & Installation

### Requirements

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (optional)
- Git

### Installation

Using `pip`:
```bash
pip install -r requirements.txt
```

Using `conda`:
```bash
conda env create -f environment.yaml
conda activate tahoe-cvae
```

---

## Dataset

We use a subset (~1M cells) from the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M), which includes:

- Single-cell gene expression matrices
- Drug perturbation metadata
- Cell line and sample metadata

Download and extract the data into the `data/` directory before running preprocessing scripts or training.

---

## Usage

### Interactive Workflow

1. Run `notebooks/loading_data.ipynb` to preprocess and store data
2. Run `notebooks/model_dev.ipynb` to train and evaluate the model

### Script Workflow

1. Train model:
   ```bash
   python src/train.py
   ```

2. Predict using trained model:
   ```bash
   python src/predict.py
   ```

---

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create your feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to your branch:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---

## License

This project is licensed under the GNU General Public License. See the `LICENSE` file for more details.

---

## Acknowledgments

- [Tahoe-100M Dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
- Inspired by the CVAE architecture from scVI and scGen
- Thanks to all contributors and open-source maintainers
