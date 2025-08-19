# Stock Prediction Model

This repository contains a Jupyter Notebook for stock price trend analysis and prediction using Python. The notebook demonstrates data collection, preprocessing, visualization, and the application of machine learning models to predict stock prices. This model is trained on nifty 100 stocks.

## Features
- Data loading and preprocessing
- Exploratory data analysis (EDA) with visualizations
- Feature engineering
- Implementation of machine learning models for stock price prediction
- Model evaluation and visualization of results

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see below)

### Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd trend_analysis
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Or install the main dependencies manually:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly yfinance textblob requests datasets transformers
   ```

### Usage
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `StockPrediction_LSTM_v2.ipynb` and run the cells sequentially.

## File Structure
- `StockPrediction_LSTM_v2.ipynb`: Main notebook for stock trend analysis and prediction.

## Example Output
The notebook provides:
- Model performance metrics (e.g., RMSE, MAE)
- Predicted value for any given stock from the symbols

