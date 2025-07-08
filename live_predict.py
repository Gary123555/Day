import pandas as pd
import yfinance as yf
import joblib
import pandas_ta as ta
from datetime import datetime
import pytz # 引入時區處理函式庫

# --- 設定區 ---
TICKER = 'TSLA'
MODEL_FILENAME = 'tsla_model.joblib'
MARKET_TIMEZONE = 'US/Eastern' # 美國東部時區

# ... (其他 FEATURES 和函式定義不變) ...
FEATURES = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD_line',
    'MACD_signal', 'MACD_hist', 'DMP', 'DMN', 'ADX', 'VWAP', 'MA_short',
    'MA_long', 'BB_mid', 'BB_std', 'BB_upper', 'BB_lower', 'BB_width'
]

def is_market_open():
    """檢查現在是否為美國股市的開盤時間"""
    tz = pytz.timezone(MARKET_TIMEZONE)
    now_et = datetime.now(tz)
    
    # 判斷是否為週一到週五
    if now_et.weekday() >= 5: # 5=週六, 6=週日
        print(f"市場檢查：今天是週末 ({now_et.strftime('%A')})，股市休市。")
        return False
        
    # 判斷時間是否在 09:30 到 16:00 之間
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0).time()
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0).time()
    
    if market_open <= now_et.time() <= market_close:
        print(f"市場檢查：現在是美東時間 {now_et.strftime('%H:%M:%S')}，在開盤時間內。")
        return True
    else:
        print(f"市場檢查：現在是美東時間 {now_et.strftime('%H:%M:%S')}，已收盤或未開盤。")
        return False

def get_live_data(ticker_symbol):
    """使用 yfinance 獲取最新的即時股票數據"""
    print(f"正在從 yfinance 獲取 {ticker_symbol} 的最新數據...")
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="200d")
    if hist.empty:
        raise ValueError("無法獲取股票數據。")
    print("成功獲取數據。")
    return hist

def calculate_live_features(df):
    """使用 pandas_ta 即時計算所有模型需要的技術指標。"""
    print("正在計算技術指標...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    df.ta.vwap(append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.bbands(length=20, append=True)
    rename_map = {'RSI_14': 'RSI','MACD_12_26_9': 'MACD_line','MACDh_12_26_9': 'MACD_hist','MACDs_12_26_9': 'MACD_signal','DMP_14': 'DMP','DMN_14': 'DMN','ADX_14': 'ADX','VWAP_D': 'VWAP','SMA_20': 'MA_short','SMA_50': 'MA_long','BBM_20_2.0': 'BB_mid','BBL_20_2.0': 'BB_lower','BBU_20_2.0': 'BB_upper','BBB_20_2.0': 'BB_width',}
    df.rename(columns=rename_map, inplace=True)
    if 'BB_upper' in df.columns and 'BB_mid' in df.columns:
        df['BB_std'] = (df['BB_upper'] - df['BB_mid']) / 2
        print("已手動計算 'BB_std' 特徵。")
    else:
        print("警告：無法計算 'BB_std'。")
    df.dropna(inplace=True)
    print("技術指標計算完成。")
    return df

def main():
    """主執行函數"""
    # === 關鍵修改：在所有操作之前，先檢查是否開盤 ===
    if not is_market_open():
        print("腳本執行結束。")
        return # 如果未開盤，直接結束程式
    
    print(f"\n--- 開始執行雲端預測腳本 for {TICKER} ---")
    try:
        model = joblib.load(MODEL_FILENAME)
        print(f"模型 '{MODEL_FILENAME}' 載入成功。")
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 '{MODEL_FILENAME}'。")
        return

    live_data = get_live_data(TICKER)
    data_with_features = calculate_live_features(live_data)
    missing_features = [f for f in FEATURES if f not in data_with_features.columns]
    if missing_features:
        print(f"致命錯誤：缺少特徵: {missing_features}")
        return

    latest_features = data_with_features[FEATURES].iloc[[-1]]
    print("\n準備用於預測的最新特徵數據：")
    print(latest_features)

    prediction = model.predict(latest_features)
    prediction_proba = model.predict_proba(latest_features)

    print("\n--- 最終預測結果 ---")
    if prediction[0] == 1:
        print(f"📈 模型預測：下一個交易日 {TICKER} 股價可能上漲 (趨勢為 1)")
        print(f"   模型判斷為上漲的信心指數: {prediction_proba[0][1]:.2%}")
    else:
        print(f"📉 模型預測：下一個交易日 {TICKER} 股價可能下跌或盤整 (趨勢為 0)")
        print(f"   模型判斷為下跌/盤整的信心指數: {prediction_proba[0][0]:.2%}")
    print("----------------------")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n腳本執行時發生致命錯誤: {e}")