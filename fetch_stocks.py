import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_fundamentals(ticker):
    """單檔抓取毛利率與本益比"""
    try:
        info = yf.Ticker(ticker).info
        gross_margins = info.get("grossMargins")
        trailing_pe = info.get("trailingPE")
        
        gm_pct = round(gross_margins * 100, 2) if gross_margins is not None else None
        pe_val = round(float(trailing_pe), 2) if trailing_pe is not None else None
        return ticker, gm_pct, pe_val
    except Exception:
        return ticker, None, None

def calculate_ma_and_fundamentals(input_csv="stock_list.csv", output_csv="stock_ma_diff.csv", batch_size=50):
    print(f"正在讀取 {input_csv}...", flush=True)
    stock_df = pd.read_csv(input_csv)
    ticker_to_name = dict(zip(stock_df["Ticker"], stock_df["Name"]))
    tickers = stock_df["Ticker"].dropna().unique().tolist()
    
    total = len(tickers)
    print(f"步驟 1/2：開始分批下載 {total} 檔股票價格與計算均線...", flush=True)

    price_results = []
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        print(f"正在下載股價第 {i + 1} 到 {min(i + batch_size, total)} 檔...", flush=True)
        
        try:
            data = yf.download(batch, period="6mo", group_by="ticker", progress=False)
        except Exception as e:
            print(f"批次下載發生錯誤: {e}", flush=True)
            continue
            
        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data.dropna(subset=["Close"]).copy()
                else:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    df = data[ticker].dropna(subset=["Close"]).copy()
                    
                if len(df) < 60:
                    continue
                
                df["MA5"] = df["Close"].rolling(window=5).mean()
                df["MA20"] = df["Close"].rolling(window=20).mean()
                df["MA60"] = df["Close"].rolling(window=60).mean()
                
                latest = df.iloc[-1]
                latest_date = df.index[-1].strftime("%Y-%m-%d")
                close = float(latest["Close"])
                ma5 = float(latest["MA5"])
                ma20 = float(latest["MA20"])
                ma60 = float(latest["MA60"])
                
                if close == 0:
                    continue
                
                diff_5_pct = ((close - ma5) / close) * 100
                diff_20_pct = ((close - ma20) / close) * 100
                diff_60_pct = ((close - ma60) / close) * 100
                
                price_results.append({
                    "資料日期": latest_date,
                    "股票代號": ticker,
                    "股票名稱": ticker_to_name.get(ticker, ""),
                    "最新收盤價": round(close, 2),
                    "週線(5MA)": round(ma5, 2),
                    "月線(20MA)": round(ma20, 2),
                    "季線(60MA)": round(ma60, 2),
                    "週線差對收盤價(%)": round(diff_5_pct, 2),
                    "月線差對收盤價(%)": round(diff_20_pct, 2),
                    "季線差對收盤價(%)": round(diff_60_pct, 2),
                })
            except Exception as e:
                print(f"計算 {ticker} 時發生錯誤: {e}", flush=True)
                
        time.sleep(1.5)

    if not price_results:
        print("警告：未成功取得任何股票數據，請檢查網路或清單。", flush=True)
        return

    valid_tickers = [item["股票代號"] for item in price_results]
    print(f"步驟 2/2：開始並行查詢 {len(valid_tickers)} 檔股票的毛利率與本益比...", flush=True)

    fundamentals_map = {}
    # 使用 8 個執行緒並行查詢，兼顧速度與避免 Yahoo 阻擋
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_fundamentals, t): t for t in valid_tickers}
        completed = 0
        for future in as_completed(futures):
            ticker, gm, pe = future.result()
            fundamentals_map[ticker] = (gm, pe)
            completed += 1
            if completed % 100 == 0 or completed == len(valid_tickers):
                print(f"基本面查詢進度：{completed} / {len(valid_tickers)} 檔", flush=True)

    # 合併基本面數據至報表
    for item in price_results:
        gm, pe = fundamentals_map.get(item["股票代號"], (None, None))
        item["毛利率(%)"] = gm
        item["本益比(PE)"] = pe

    result_df = pd.DataFrame(price_results)
    
    # 依照「月線差對收盤價(%)」由小到大排序（負值最多者排在最前面）
    result_df = result_df.sort_values(by="月線差對收盤價(%)", ascending=True)
    
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[{datetime.now()}] 完成！共 {len(result_df)} 檔股票已儲存至 {output_csv}。", flush=True)

if __name__ == "__main__":
    calculate_ma_and_fundamentals()
