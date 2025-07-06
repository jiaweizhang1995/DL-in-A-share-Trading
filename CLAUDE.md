# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning project for Chinese A-share stock trading prediction. The model uses a hybrid neural network architecture combining fully connected layers, 1D/2D convolutions, and LSTM layers to predict stock price movements. The project is written in Chinese and processes real Chinese stock market data.

## Development Workflow

### 1. Data Collection
```bash
python Building_a_Dataset.py
```
- Uses baostock API to collect A-share stock data from 2010-2025
- Filters stocks (excludes those starting with 8, 68, 4)
- Saves data as CSV files in `./data/` directory
- Creates temporary batch files during processing

### 2. Model Training
```bash
python Training.py
```
- Trains the hybrid neural network model
- Creates train/test split (2010-2021 for training, 2022+ for testing)
- Saves best model weights to `./weights/model_baseline.pt`
- Generates training plots (loss curves, correlation plots)

### 3. Cross-Validation
```bash
python 5_fold_CV.py
```
- Performs 3-fold cross-validation (despite the filename)
- Saves multiple model files: `./weights/model_APP_0.pt`, `./weights/model_APP_1.pt`, etc.
- Generates validation plots for each fold

### 4. Model Application
```bash
python APP.py
```
- Fetches latest stock data using akshare API
- Loads all cross-validated models and ensemble predicts
- Outputs predictions to `output.csv`

## Key Dependencies

Install required packages:
```bash
pip install scikit-learn numpy pandas matplotlib tqdm akshare baostock
```

For PyTorch, visit the official website to get the appropriate version for your hardware.

## Model Architecture

The neural network combines:
- **Fully Connected Layers**: 2000 → 1000 → 100 → 1 neurons with BatchNorm and Dropout
- **1D Convolution**: 6 → 16 → 32 channels for temporal feature extraction
- **2D Convolution**: 1 → 16 → 32 channels for spatial feature extraction  
- **LSTM**: Bidirectional 4-layer LSTM for sequence modeling
- **Kaiming Initialization**: For all conv and linear layers

## Data Processing

- **Input Features**: Open, Close, High, Low, Volume, Turnover Rate (60 days)
- **Target**: Average volume 25 days ahead as percentage change
- **Filtering**: Removes stocks with < 180 trading days, excludes first 20 days after listing
- **Normalization**: Min-max normalization applied during forward pass

## File Structure

- `Building_a_Dataset.py`: Data collection script using baostock
- `Training.py`: Single model training with train/test split
- `5_fold_CV.py`: Cross-validation training (actually 3-fold)
- `APP.py`: Real-time prediction application using akshare
- `data/`: Contains CSV files with stock data
- `weights/`: Stores trained model weights (.pt files)

## Important Notes

- The code uses Chinese variable names and comments
- Data sources: baostock (historical) and akshare (real-time)
- Model predicts volume-based percentage changes, not direct price movements
- Cross-validation uses 3 folds despite filename suggesting 5
- GPU acceleration available if CUDA is detected