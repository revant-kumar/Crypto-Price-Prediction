# Cryptocurrency Price Prediction using LSTM

This project aims to predict the price of Cryptocurrency (ex: Bitcoin (BTC-USD)) using Long Short-Term Memory (LSTM) neural networks. The model is built using TensorFlow and Keras, and the dataset is fetched from Yahoo Finance using the `yfinance` library. The model is trained on historical data and used to predict future prices. This model takes the records from last 60 days to predict the price of the crypto on the next day.

## Table of Contents

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Price Prediction](#price-prediction)
- [Results](#results)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project leverages LSTM, a type of recurrent neural network (RNN), to predict Bitcoin's price based on historical data. LSTM is particularly effective for time series prediction tasks due to its ability to retain information over long sequences, making it ideal for financial data modeling.

The project follows these steps:
1. Fetch historical Bitcoin data using `yfinance`.
2. Preprocess the data by scaling it between 0 and 1.
3. Split the data into training and testing sets.
4. Build and train the LSTM model.
5. Test the model on unseen data and visualize the predictions.
6. Predict future Bitcoin prices based on real-time data.

## Dependencies

To run this project, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `pandas_datareader`
- `scikit-learn`
- `tensorflow`

You can install all dependencies using the following command:

```bash
pip install numpy pandas matplotlib yfinance pandas_datareader scikit-learn tensorflow
```

## Data Collection

The historical price data of Bitcoin (BTC-USD) is collected from Yahoo Finance using the `yfinance` library. The dataset spans from January 1, 2016, to the current date.

```python
import yfinance as yf
import datetime as dt

crypto_currency = 'BTC-USD'
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = yf.download(crypto_currency, start=start, end=end)
```

## Data Preprocessing

The `Close` prices are scaled between 0 and 1 using `MinMaxScaler` for better performance during training. The data is then split into sequences, where each sequence is 60 days long, and the model is trained to predict the price on the 61st day.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
```

## Model Architecture

The LSTM model consists of the following layers:

- Three LSTM layers with 50 units each.
- Dropout layers to prevent overfitting.
- A Dense layer with a single unit to output the predicted price.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
```

## Training

The model is trained on the historical data using 25 epochs and a batch size of 32.

```python
model.fit(x_train, Y_train, epochs=25, batch_size=32)
```

## Testing and Evaluation

The model is tested on the data from January 2024 to the current date. The predicted prices are compared with the actual prices, and the results are visualized using `matplotlib`.

```python
test_start = dt.datetime(2024, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download(crypto_currency, start=test_start, end=test_end)
```

## Price Prediction

The trained model is used to predict future Bitcoin prices. The most recent data is fed into the model, and the predicted price for the next day is output.

```python
prediction = model.predict(real_data)
```

## Results

The model's predictions are visualized alongside the actual Bitcoin prices. The chart shows the effectiveness of the LSTM model in capturing the trends and predicting future prices.

![image](https://github.com/user-attachments/assets/7ae8f345-8300-4832-80c2-2ada579711e2)


## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-price-prediction.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python crypto_price_prediction.py
```

4. The model will fetch the data, train the LSTM model, and output the predicted prices.

## Contributing

Contributions are welcome! If you have suggestions for improving this project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
