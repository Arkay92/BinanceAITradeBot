import concurrent.futures
from functools import partial
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
from ta import add_all_ta_features
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.websockets import BinanceSocketManager
import logging
import math
import time

# Configuration
class Config:
    API_KEY = 'your_api_key'  # Replace with your actual API key
    API_SECRET = 'your_api_secret'  # Replace with your actual API secret
    CRYPTO_ID = 'bitcoin'
    CRYPTO_SYMBOL = 'BTCUSDT'
    DAYS = 'max'
    INTERVAL = 'daily'
    BUY_THRESHOLD = 0.7  # Adjusted based on analysis
    SELL_THRESHOLD = 0.3  # Adjusted based on analysis
    RISK_FACTOR = 0.02  # Adjusted for potentially higher returns
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    LOT_SIZE_ROUNDING = 8
    TEXTS = ["Bitcoin hits new all-time high", "Ethereum upgrade introduces deflationary mechanism"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API Setup
client = Client(Config.API_KEY, Config.API_SECRET)

# Function to fetch crypto prices with indicators
def fetch_crypto_prices_with_indicators():
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{Config.CRYPTO_ID}/market_chart'
        params = {'vs_currency': 'usd', 'days': Config.DAYS, 'interval': Config.INTERVAL}
        response = requests.get(url, params=params)
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop(columns='timestamp')
        df = add_all_ta_features(df, open="price", high="price", low="price", close="price", volume="price", fillna=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

# Function to get combined embeddings for texts
def get_combined_embeddings(texts):
    try:
        tokenizer_bitcoin = AutoTokenizer.from_pretrained("huggingtweets/bitcoin")
        model_bitcoin = AutoModel.from_pretrained("huggingtweets/bitcoin")
        tokenizer_cryptobert = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
        model_cryptobert = AutoModel.from_pretrained("ElKulako/cryptobert")
        combined_embeddings = []
        
        # Function to generate embedding for a single text
        def generate_embedding(text, tokenizer, model):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Generate embeddings concurrently for each text
            futures = [executor.submit(partial(generate_embedding, text, tokenizer_bitcoin, model_bitcoin)) for text in texts]
            bitcoin_embeddings = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            futures = [executor.submit(partial(generate_embedding, text, tokenizer_cryptobert, model_cryptobert)) for text in texts]
            cryptobert_embeddings = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Combine embeddings
        combined_embeddings = [np.concatenate((bitcoin_emb, cryptobert_emb)) for bitcoin_emb, cryptobert_emb in zip(bitcoin_embeddings, cryptobert_embeddings)]
        return np.array(combined_embeddings)
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return np.empty((0,))

# Class for Crypto Hybrid Model
class CryptoHybridModel(nn.Module):
    def __init__(self):
        super(CryptoHybridModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Function to preprocess data
def preprocess_data(msg, scaler):
    try:
        price = float(msg['p'])
        scaled_price = scaler.transform([[price]])
        return torch.tensor(scaled_price, dtype=torch.float).unsqueeze(0)
    except KeyError as e:
        logging.error(f"Key error in message data: {e}")
        return torch.zeros(1, 1, dtype=torch.float)

# Function to calculate quantity
def calculate_quantity(symbol='BTCUSDT', risk_factor=Config.RISK_FACTOR):
    try:
        account_details = client.get_account()
        free_balance = next((item for item in account_details['balances'] if item["asset"] == symbol[:-4]), None)['free']
        free_balance = float(free_balance)
        quantity = free_balance * risk_factor
        return adjust_quantity_to_meet_requirements(quantity, symbol)
    except Exception as e:
        logging.error(f"Error calculating quantity: {e}")
        return 0

# Function to adjust quantity to meet requirements
def adjust_quantity_to_meet_requirements(quantity, symbol='BTCUSDT'):
    try:
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((filter for filter in symbol_info['filters'] if filter['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            min_qty = float(lot_size_filter['minQty'])
            max_qty = float(lot_size_filter['maxQty'])
            step_size = float(lot_size_filter['stepSize'])

            quantity = max(min_qty, min(quantity, max_qty))
            step_size_scale = round(-math.log(step_size, 10))
            quantity = round(quantity / step_size) * step_size
            quantity = round(quantity, Config.LOT_SIZE_ROUNDING)

            return quantity
        else:
            logging.error(f"LOT_SIZE filter not found for symbol: {symbol}")
            return 0
    except Exception as e:
        logging.error(f"Error adjusting quantity: {e}")
        return 0

# Function to process trade message
def process_message(msg, model, scaler):
    try:
        if msg['e'] == 'trade':
            data = preprocess_data(msg, scaler)
            prediction = model(data)
            if prediction.item() > Config.BUY_THRESHOLD:
                # Adjusted to potentially buy more aggressively
                order = client.order_market_buy(symbol=Config.CRYPTO_SYMBOL, quantity=calculate_quantity(Config.CRYPTO_SYMBOL, risk_factor=Config.RISK_FACTOR * 2))
                logging.info(f"Buy order placed: {order}")
            elif prediction.item() < Config.SELL_THRESHOLD:
                # Adjusted to potentially sell more aggressively
                order = client.order_market_sell(symbol=Config.CRYPTO_SYMBOL, quantity=calculate_quantity(Config.CRYPTO_SYMBOL, risk_factor=Config.RISK_FACTOR * 2))
                logging.info(f"Sell order placed: {order}")
    except (BinanceAPIException, BinanceOrderException) as e:
        logging.error(f"Binance error: {e}")
    except Exception as e:
        logging.error(f"Error processing message: {e}")

# Function to train the model
def train_model(model, X_train, y_train):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)

    for epoch in range(Config.NUM_EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    criterion = nn.MSELoss()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float).unsqueeze(1)
    predictions = model(X_test_tensor)
    mse = criterion(predictions, torch.tensor(y_test, dtype=torch.float).unsqueeze(1))
    logging.info(f'Test MSE: {mse.item()}')

if __name__ == "__main__":
    # Binance WebSocket Setup
    bm = BinanceSocketManager(client)
    bm.start_trade_socket(Config.CRYPTO_SYMBOL, lambda msg: process_message(msg, model, scaler))
    bm.start()

    # Fetch price data and preprocess
    price_data = fetch_crypto_prices_with_indicators()
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(price_data.drop(columns=['date']))

    # Generate combined embeddings for text data
    combined_embeddings = get_combined_embeddings(Config.TEXTS)

    # Combine features and split data for training and testing
    combined_features = np.hstack((normalized_data[:-1], combined_embeddings[:-1]))
    X_train, X_test, y_train, y_test = train_test_split(combined_features, normalized_data[1:, 0], test_size=0.2, random_state=42)

    # Define and train the model
    model = CryptoHybridModel()
    train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
