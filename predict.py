import os
import pandas as pd
import yfinance as yf
import joblib
import requests
import pandas_ta as ta

# --- 設定區 ---
WEBHOOK_URL = os.getenv('TRADINGVIEW_WEBHOOK_URL')
TICKER = 'TSLA'
MODEL_FILENAME = 'tsla_model.joblib'

# ================== 關鍵修改 ==================
# 和您本地訓練時完全一致的特徵列表
FEATURES = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD_line',
    'MACD_signal', 'MACD_hist', 'DMP', 'DMN', 'ADX', 'VWAP', 'MA_short',
    'MA_long', 'BB_mid', 'BB_std', 'BB_upper', 'BB_lower', 'BB_width'
]
# ===============================================

def get_latest_data(ticker_symbol):
    """獲取最新的股票數據"""
    print(f"正在從 yfinance 獲取 {ticker_symbol} 的最新數據...")
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="250d") # 獲取足夠數據來計算指標
    if hist.empty:
        raise ValueError("無法獲取股票數據。")
    print("成功獲取數據。")
    return hist

def calculate_features(df):
    """計算所有需要的技術指標"""
    print("正在計算技術指標...")
    df.ta.rsi(length=14, append=True) # RSI_14
    df.ta.macd(append=True) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    df.ta.adx(append=True) # ADX_14, DMP_14, DMN_14
    df.ta.vwap(append=True) # VWAP_D
    df.ta.sma(length=20, append=True) # SMA_20
    df.ta.sma(length=50, append=True) # SMA_50
    df.ta.bbands(length=20, append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    
    # 為了匹配特徵列表，我們需要重新命名欄位
    rename_map = {
        'RSI_14': 'RSI',
        'MACD_12_26_9': 'MACD_line',
        'MACDh_12_26_9': 'MACD_hist',
        'MACDs_12_26_9': 'MACD_signal',
        'DMP_14': 'DMP',
        'DMN_14': 'DMN',
        'ADX_14': 'ADX',
        'VWAP_D': 'VWAP',
        'SMA_20': 'MA_short',
        'SMA_50': 'MA_long',
        'BBM_20_2.0': 'BB_mid',
        'BBL_20_2.0': 'BB_lower',
        'BBU_20_2.0': 'BB_upper',
        'BBB_20_2.0': 'BB_width',
        'BBP_20_2.0': 'BB_std' # 使用 BBP (布林帶寬度百分比) 作為 BB_std
    }
    df.rename(columns=rename_map, inplace=True)
    df.dropna(inplace=True)
    print("技術指標計算完成。")
    return df

def main():
    print("開始執行預測腳本...")
    model = joblib.load(MODEL_FILENAME)
    print("模型載入成功。")
    
    data = get_latest_data(TICKER)
    data_with_features = calculate_features(data)
    
    missing_features = [f for f in FEATURES if f not in data_with_features.columns]
    if missing_features:
        print(f"錯誤：計算後，數據中缺少模型需要的特徵: {missing_features}")
        return

    features_for_prediction = data_with_features[FEATURES].iloc[[-1]]
    
    print("準備用於預測的特徵數據：")
    print(features_for_prediction)
    
    prediction = model.predict(features_for_prediction)[0]
    print(f"模型預測結果: {prediction}")
    
    if str(prediction) == '1' or str(prediction).lower() == 'buy':
        print("偵測到買入訊號，準備發送通知...")
        # ... (省略發送 Webhook 的程式碼) ...
    else:
        print(f"模型預測結果為 '{prediction}'，未觸發買入訊號。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"腳本執行時發生致命錯誤: {e}")
