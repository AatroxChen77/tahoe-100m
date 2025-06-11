# Project Name

## Overview
This project contains a collection of Jupyter notebooks for data preparation, model training, and results visualization. The notebooks are designed to work with the Tahoe-100M dataset, which contains single-cell gene expression data. We are looking into the subset of 1M of this dataset and use conditional Variational Auto-encoder to model gene expression changes associated with drug perturbaitons.

## Project Structure

### Notebooks

1. **loading_data.ipynb**
   - Purpose: Data preprocessing and preparation
   - Main tasks:
     - Import required libraries
     - Load and inspect the data
     - Data cleaning and preprocessing
     - Feature engineering
     - Data validation
     - Save processed data

2. **model_dev.ipynb**
   - Purpose: Model development and training
   - Main tasks:
     - Load preprocessed data
     - Model architecture definition
     - Training pipeline implementation
     - Model evaluation
     - Model saving
     -Visualization of the training curve
     -Analysis of predicted genes in hold-out set
     -Best models are saved in models folder


## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install -r requirements.txt
```

or 

```bash
conda env create -f enviroment.yaml
```




## Usage

1. Start with `loading_data.ipynb` to preprocess your data
2. Run `model_dev.ipynb` to train your models

## Data

The project uses the Tahoe-100M dataset (https://huggingface.co/datasets/tahoebio/Tahoe-100M), which contains:
- Single-cell gene expression data
- Drug treatment information
- Cell line metadata
- Sample metadata

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU License - see the LICENSE file for details.

## Acknowledgments

- Tahoe-100M dataset providers (https://huggingface.co/datasets/tahoebio/Tahoe-100M)
- Contributors and maintainers