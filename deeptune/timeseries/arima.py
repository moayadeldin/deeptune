from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from sklearn.metrics import mean_squared_error
import options
from deeptune.utilities import get_args
import matplotlib.pyplot as plt

parser = options.parser
args = get_args()

INPUT_DIR = args.input_dir
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
TARGET_COLUMN = args.target_column
order = (2,1,2) # default hyperparameters choice for arima

def load_data(file_path):
    
    df = pd.read_parquet(file_path)
    return df

def train_test_split(df, test_size):
    target_col = TARGET_COLUMN
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce') 
    df = df.dropna(subset=[target_col])

    train_size = int(len(df) * (1 - test_size))
    train = df[target_col].iloc[:train_size]
    test = df[target_col].iloc[train_size:]
    return train, test

def train_arima_model(train_data,order=(1, 1, 1)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def forecast(model_fit, test_data):
    steps = len(test_data)  # forecast the full test range
    forecast = model_fit.forecast(steps=steps)
    mse = mean_squared_error(test_data, forecast)
    print(f"MSE Value Is: {mse:.4f}")
    return forecast

def plot_forecast(train, test, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual", color="gray")
    plt.plot(test.index, forecast, label="Forecast", color="red")
    plt.title("ARIMA Forecast vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data(INPUT_DIR)
    
    train_data, test_data = train_test_split(df, TEST_SIZE)
    
    model_fit = train_arima_model(train_data)
    
    forecast_value = forecast(model_fit, test_data)
    
    print(f"Forecasted Value: {forecast_value}")

    plot_forecast(train_data, test_data, forecast_value)

