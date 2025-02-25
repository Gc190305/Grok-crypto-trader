
import tkinter as tk
from tkinter import scrolledtext, messagebox
import requests
import pandas as pd
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import threading
import re

# Configuration
TRADING_PARAMS = {
    "priority_fee": 0.0001,  # SOL
    "buy_amount": 0.1,       # SOL
    "slippage": 5,           # Percentage
    "profit_targets": [2, 3] # Sell at 2x, 3x
}

# API Keys (Replace with your own)
API_KEYS = {
    "pump_fun": "YOUR_PUMP_FUN_API_KEY",
    "dex_screener": "YOUR_DEX_SCREENER_API_KEY",
    "rugcheck": "YOUR_RUGCHECK_API_KEY",
    "gmgn": "YOUR_GMGN_API_KEY"
}

# Top Crypto Influencers on X
X_KOLS = [
    "@VitalikButerin",   # 7.2M followers
    "@cz_binance",       # 8.9M followers
    "@brian_armstrong",  # 1.3M followers
    "@elonmusk",         # 200M+ followers
    "@APompliano"        # 1.7M followers
]

# Fetch Functions
def fetch_pump_fun_data(log):
    log.insert(tk.END, "Fetching Pump.fun data...\n")
    url = "https://api.pump.fun/tokens"
    headers = {"Authorization": f"Bearer {API_KEYS['pump_fun']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        log.insert(tk.END, "Pump.fun data fetched.\n")
        return response.json()
    log.insert(tk.END, "Failed to fetch Pump.fun data.\n")
    return None

def fetch_dex_screener_data(token_address, log):
    log.insert(tk.END, f"Fetching DexScreener data for {token_address}...\n")
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    response = requests.get(url)
    if response.status_code == 200:
        log.insert(tk.END, "DexScreener data fetched.\n")
        return response.json()
    log.insert(tk.END, "Failed to fetch DexScreener data.\n")
    return None

def fetch_rugcheck_data(contract_address, log):
    log.insert(tk.END, f"Running RugCheck on {contract_address}...\n")
    url = f"https://api.rugcheck.xyz/v1/contracts/{contract_address}"
    headers = {"Authorization": f"Bearer {API_KEYS['rugcheck']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        score = data.get("contract_score", 0)
        log.insert(tk.END, f"RugCheck score: {score}\n")
        return data if score >= 50 else None
    log.insert(tk.END, "Failed to fetch RugCheck data.\n")
    return None

def fetch_gmgn_data(token_address, log):
    log.insert(tk.END, f"Fetching GmGn data for {token_address}...\n")
    url = f"https://api.gmgn.ai/v1/tokens/{token_address}"
    headers = {"Authorization": f"Bearer {API_KEYS['gmgn']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        log.insert(tk.END, "GmGn data fetched.\n")
        return response.json()
    log.insert(tk.END, "Failed to fetch GmGn data.\n")
    return None

# X Post Scanner (Simulated New Solana Meme Coins)
def scan_x_for_cas(log):
    log.insert(tk.END, "Scanning X posts from top crypto KOLs for new Solana meme coins...\n")
    cas = []
    for kol in X_KOLS:
        log.insert(tk.END, f"Checking posts from {kol}...\n")
        # Simulated posts with new Solana meme coins (Feb 2025 trends)
        sample_posts = []
        if kol == "@VitalikButerin":
            sample_posts = ["Exploring Solana’s meme scene: SOLMEME1234567890ABCDEFGHIJ"]
        elif kol == "@cz_binance":
            sample_posts = ["Binance might list this: PUMPIT4567890ABCDEFGHIJKLMN"]
        elif kol == "@brian_armstrong":
            sample_posts = ["New Solana gem: COINCAT7890ABCDEFGHIJKLMNOPQ"]
        elif kol == "@elonmusk":
            sample_posts = ["To the moon with: MOONBOY1234567890ABCDEFGHIJK"]
        elif kol == "@APompliano":
            sample_posts = ["This Solana meme coin’s hot: BULLRUN4567890ABCDEFGH"]

        for post in sample_posts:
            ca_match = re.search(r'[A-Za-z0-9]{43,44}', post)
            if ca_match:
                ca = ca_match.group(0)
                log.insert(tk.END, f"Found CA {ca} from {kol}\n")
                if fetch_rugcheck_data(ca, log):
                    cas.append(ca)
                else:
                    log.insert(tk.END, f"CA {ca} flagged as potential scam.\n")
    return cas

# Preprocessing
def preprocess_data(pump_data, dex_data, gmgn_data, log):
    log.insert(tk.END, "Preprocessing data...\n")
    pump_df = pd.DataFrame(pump_data)
    dex_df = pd.DataFrame(dex_data.get("pairs", []))
    gmgn_df = pd.DataFrame([gmgn_data])
    merged_df = pd.merge(pump_df, dex_df, on="token_address", how="inner")
    merged_df = pd.merge(merged_df, gmgn_df, on="token_address", how="inner")
    merged_df.fillna(0, inplace=True)
    scaler = StandardScaler()
    numerical_features = ["price", "volume", "liquidity"]
    merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])
    log.insert(tk.END, "Data preprocessed.\n")
    return merged_df

def train_model(data, log):
    log.insert(tk.END, "Training model...\n")
    X = data[["price", "volume", "liquidity"]]
    y = data["profitable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    log.insert(tk.END, f"Model trained. Accuracy: {accuracy}\n")
    return model

# Monitor and Decide
def monitor_and_decide(token_address, buy_price, data, model, log):
    current_price = float(data["pairs"][0].get("priceUsd", 0)) if data["pairs"] else 0
    log.insert(tk.END, f"Monitoring {token_address}: Current price ${current_price}\n")
    if current_price > 0:
        log.insert(tk.END, f"Would buy {token_address} at ${buy_price}\n")
        for target in TRADING_PARAMS["profit_targets"]:
            if current_price >= buy_price * target:
                log.insert(tk.END, f"Would sell {token_address} at {target}x profit (${current_price})\n")
                return True
    return False

# Main Bot Logic
def run_bot(log):
    async def bot_logic():
        log.insert(tk.END, "Hey there, dude! Starting the crypto hunt...\n")
        cas = scan_x_for_cas(log)
        if not cas:
            log.insert(tk.END, "No safe Solana meme CAs found in X posts.\n")
            return

        for token_address in cas:
            pump_data = fetch_pump_fun_data(log)
            dex_data = fetch_dex_screener_data(token_address, log)
            gmgn_data = fetch_gmgn_data(token_address, log)
            if not all([pump_data, dex_data, gmgn_data]):
                continue

            processed_data = preprocess_data(pump_data, dex_data, gmgn_data, log)
            processed_data["profitable"] = (processed_data["price"] > 0).astype(int)
            model = train_model(processed_data, log)

            buy_price = float(dex_data["pairs"][0].get("priceUsd", 0)) if dex_data["pairs"] else 0
            if buy_price > 0:
                monitor_and_decide(token_address, buy_price, dex_data, model, log)

        log.insert(tk.END, "Bot execution complete. Catch ya later!\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bot_logic())

def start_bot_thread(log):
    thread = threading.Thread(target=run_bot, args=(log,))
    thread.start()

# GUI Setup
root = tk.Tk()
root.title("Grok’s Crypto Trading Bot")
root.geometry("600x400")

greeting_label = tk.Label(root, text="Yo, dude! Welcome to Grok’s Crypto Trader!", font=("Arial", 14))
greeting_label.pack(pady=5)

log = scrolledtext.ScrolledText(root, width=70, height=20)
log.pack(pady=10)

start_button = tk.Button(root, text="Start Bot", command=lambda: start_bot_thread(log))
start_button.pack(pady=5)

root.mainloop()
