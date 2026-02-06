from flask import Flask, request, jsonify
from metaapi_cloud_sdk import MetaApi
import asyncio
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from waitress import serve
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
# If credentials are not passed in request, fallback to env vars (Admin Mode)
DEFAULT_TOKEN = os.getenv("METAAPI_TOKEN")
DEFAULT_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")
N8N_SIGNAL_URL = os.getenv("N8N_WEBHOOK_URL")

# Trading Settings
RISK_REWARD_RATIO = 3.0
Z_SCORE_THRESHOLD = 2.0
LOOKBACK_BARS = 50
WATCHLIST = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
    "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD", "BTCUSD"
]

# --- HELPER FUNCTIONS ---

async def get_mt_connection(token, account_id):
    """
    Connects to MetaApi using the provided token and account ID.
    """
    if not token or not account_id:
        return None, None, "Missing credentials"
    
    try:
        api = MetaApi(token=token)
        account = await api.metatrader_account_api.get_account(account_id)
        
        # Ensure account is deployed and connected
        if account.state != 'DEPLOYED':
            print(f"Deploying account {account_id}...")
            await account.deploy()
        
        print(f"Waiting for connection to {account_id}...")
        await account.wait_connected()
        
        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()
        
        return api, connection, None
    except Exception as e:
        return None, None, str(e)

def calculate_quant_metrics(df):
    """
    Same Logic: Z-Score & Volatility
    """
    closes = df['close'].values
    mean = np.mean(closes)
    std_dev = np.std(closes)
    current_price = closes[-1]
    
    if std_dev == 0:
        return 0, 0
    
    z_score = (current_price - mean) / std_dev
    return z_score, std_dev

# --- ROUTES ---

@app.route('/connect-account', methods=['POST'])
def connect_account():
    """
    Validates credentials sent from Frontend.
    """
    data = request.json
    token = data.get('token')
    account_id = data.get('account_id')
    
    async def _test():
        api, connection, error = await get_mt_connection(token, account_id)
        if error:
            return jsonify({"status": "error", "message": error}), 400
        
        # Get Account Information
        account_info = await connection.get_account_information()
        return jsonify({
            "status": "connected", 
            "broker": account_info.get('broker'),
            "currency": account_info.get('currency'),
            "balance": account_info.get('balance')
        })

    return asyncio.run(_test())

@app.route('/force-scan', methods=['POST'])
def force_scan():
    """
    Scans the market using the Z-Score strategy.
    Expects credentials in body, or falls back to ENV.
    """
    data = request.json or {}
    token = data.get('token', DEFAULT_TOKEN)
    account_id = data.get('account_id', DEFAULT_ACCOUNT_ID)
    
    print(f"--- STARTING CLOUD SCAN {datetime.now()} ---")

    async def _scan():
        api, connection, error = await get_mt_connection(token, account_id)
        if error:
            return jsonify({"status": "error", "message": error}), 400

        best_signal = None
        highest_magnitude = 0

        for symbol in WATCHLIST:
            try:
                # Get Candles (MetaApi format: '15m')
                # Start time logic: Look back enough to get N bars
                # Easier way: get_candidates implies getting history
                # We fetch last 100 candles to be safe
                history = await connection.get_historical_candles(symbol, '15m', datetime.now(), 0, LOOKBACK_BARS + 10)
                
                if not history:
                    continue
                
                # Convert to DataFrame
                # MetaApi returns list of dicts: {'time':..., 'open':..., 'close':...}
                df = pd.DataFrame(history)
                df['close'] = df['close'].astype(float)
                
                # --- REASONING ---
                z_score, volatility = calculate_quant_metrics(df)
                abs_z = abs(z_score)

                if abs_z > Z_SCORE_THRESHOLD:
                    action = "BUY" if z_score > 0 else "SELL" # Momentum
                    current_price = df['close'].iloc[-1]
                    
                    risk_distance = volatility * 1.5
                    
                    if action == "BUY":
                        sl = current_price - risk_distance
                        tp = current_price + (risk_distance * RISK_REWARD_RATIO)
                    else:
                        sl = current_price + risk_distance
                        tp = current_price - (risk_distance * RISK_REWARD_RATIO)

                    # Get Symbol Info for digits
                    symbol_info = await connection.get_symbol_specification(symbol)
                    digits = symbol_info.get('digits', 5)
                    sl = round(sl, digits)
                    tp = round(tp, digits)
                    
                    confidence = min(0.99, abs_z / 4.0)

                    if abs_z > highest_magnitude:
                        highest_magnitude = abs_z
                        best_signal = {
                            "symbol": symbol,
                            "direction": action,
                            "volume": 0.1,
                            "entry_price": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "risk_reward": RISK_REWARD_RATIO,
                            "confidence": float(confidence),
                            "reasoning": f"Cloud Z-Score {z_score:.2f} (> {Z_SCORE_THRESHOLD})"
                        }

            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue

        # Clean up
        # await connection.close() # Optional, but good practice if not reusing

        if best_signal:
            print(f"SIGNAL FOUND: {best_signal['symbol']} {best_signal['direction']}")
            # Send to N8N
            if N8N_SIGNAL_URL:
                import requests
                try:
                    requests.post(N8N_SIGNAL_URL, json=best_signal)
                except Exception as e:
                    print(f"N8N Error: {e}")
            
            return jsonify({"status": "signal_sent", "data": best_signal}), 200
        else:
            return jsonify({"status": "no_trade", "message": "No specific anomalies."}), 200

    return asyncio.run(_scan())

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """
    Executes a trade on the Cloud Account.
    """
    data = request.json
    token = data.get('token', DEFAULT_TOKEN)
    account_id = data.get('account_id', DEFAULT_ACCOUNT_ID)
    
    symbol = data.get('symbol')
    action = data.get('direction') or data.get('action') 
    volume = float(data.get('volume', 0.01))
    sl = float(data.get('stop_loss', 0))
    tp = float(data.get('take_profit', 0))

    if not symbol or not action:
        return jsonify({"status": "error", "message": "Missing symbol/action"}), 400

    async def _trade():
        api, connection, error = await get_mt_connection(token, account_id)
        if error:
            return jsonify({"status": "error", "message": error}), 400
        
        try:
            order_type = 'ORDER_TYPE_BUY' if action == "BUY" else 'ORDER_TYPE_SELL'
            
            # MetaApi simplified trading
            if action == 'BUY':
                result = await connection.create_market_buy_order(
                    symbol=symbol, 
                    volume=volume, 
                    stop_loss=sl, 
                    take_profit=tp,
                    options={"comment": "Cloud-Quant-Bot"}
                )
            else:
                result = await connection.create_market_sell_order(
                    symbol=symbol, 
                    volume=volume, 
                    stop_loss=sl, 
                    take_profit=tp,
                    options={"comment": "Cloud-Quant-Bot"}
                )
            
            return jsonify({"status": "executed", "ticket": result.get('orderId')}), 200
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return asyncio.run(_trade())

if __name__ == '__main__':
    print("--- CLOUD QUANT ENGINE ONLINE ---")
    serve(app, host='0.0.0.0', port=5000)