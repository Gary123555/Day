import pandas as pd
import joblib

# 定義特徵欄位 (必須與訓練時完全一致)
FEATURES = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD_line',
    'MACD_signal', 'MACD_hist', 'DMP', 'DMN', 'ADX', 'VWAP', 'MA_short',
    'MA_long', 'BB_mid', 'BB_std', 'BB_upper', 'BB_lower', 'BB_width'
]

def make_prediction():
    """
    載入模型和最新數據，並對下一個交易日進行預測。
    """
    try:
        # 載入訓練好的模型
        model = joblib.load('tsla_model.joblib')
        print("模型 'tsla_model.joblib' 載入成功。")
    except FileNotFoundError:
        print("錯誤：找不到模型文件 'tsla_model.joblib'。請先運行訓練並保存模型的腳本。")
        return

    try:
        # 讀取數據
        data = pd.read_csv('TSLA_data_labeled.csv', parse_dates=['Date'])
        print("數據 'TSLA_data_labeled.csv' 載入成功。")
    except FileNotFoundError:
        print("錯誤：找不到數據文件 'TSLA_data_labeled.csv'。")
        return

    # 選取最後一筆數據作為預測的基礎
    latest_data = data.iloc[[-1]]
    latest_date = latest_data['Date'].dt.date.iloc[0]
    
    # 確保只使用訓練時的特徵
    features_for_prediction = latest_data[FEATURES]

    print(f"\n使用日期為 {latest_date} 的數據進行預測...")

    # 進行預測
    prediction = model.predict(features_for_prediction)
    prediction_proba = model.predict_proba(features_for_prediction)

    # 顯示結果
    print("\n--- 最終預測結果 ---")
    if prediction[0] == 1:
        print("📈 模型預測：下一個交易日股價可能上漲 (趨勢為 1)")
        print(f"   模型判斷為上漲的信心指數: {prediction_proba[0][1]:.2%}")
    else:
        print("📉 模型預測：下一個交易日股價可能下跌或盤整 (趨勢為 0)")
        print(f"   模型判斷為下跌/盤整的信心指數: {prediction_proba[0][0]:.2%}")
    print("----------------------")


if __name__ == "__main__":
    make_prediction()
