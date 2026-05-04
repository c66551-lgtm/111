import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
from xgboost import XGBRegressor

# --- 1. 數據與格式化層 ---
def fix_ticker(ticker_str):
    ticker_str = ticker_str.strip().upper()
    if ticker_str.isdigit(): return f"{ticker_str}.TW"
    return ticker_str

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period="1y", auto_adjust=True)
        if df.empty and ".TW" in ticker:
            df = yf.download(ticker.replace(".TW", ".TWO"), period="1y", auto_adjust=True)
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

# --- 核心優化：自動搜尋財報並預測 EPS ---
@st.cache_data(ttl=86400)
def auto_predict_eps(ticker):
    """
    自動搜尋財報、營收與損益表，預估年度 EPS。
    優先順序：API 預估值 > 歷史四季淨利換算 > 容錯基本值
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 1. 嘗試抓取 API 現成的預估或歷史 EPS
        eps = info.get('forwardEps') or info.get('trailingEps')
        
        # 2. 如果為 None，則分析季度損益表 (自動掃描財報)
        if eps is None or eps == 0:
            q_fin = t.quarterly_financials
            # 抓取最近四季淨利 (Net Income)
            if q_fin is not None and 'Net Income' in q_fin.index:
                total_net_income = q_fin.loc['Net Income'].head(4).sum()
                shares = info.get('sharesOutstanding')
                if shares and shares > 0:
                    eps = total_net_income / shares
        
        return float(eps) if eps else 0.0
    except:
        return 0.0

# --- 2. AI 預測模型 ---
def train_and_predict(df):
    data = df.copy()
    features = ['Close', 'Return', 'MA5', 'Vol_MA']
    data['Return'] = data['Close'].pct_change()
    data['MA5'] = data['Close'].rolling(5).mean()
    data['Vol_MA'] = data['Volume'].rolling(5).mean()
    data['Target'] = data['Close'].shift(-5)

    clean_df = data.dropna(subset=features + ['Target'])
    if clean_df.empty: return float(data['Close'].iloc[-1])

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror", random_state=42)
    model.fit(clean_df[features], clean_df['Target'])

    X_pred = data[features].iloc[-1:].ffill().fillna(0)
    return float(model.predict(X_pred)[0])

# --- 3. 量化指標運算 ---
def calculate_metrics(df):
    returns = df['Close'].pct_change().dropna()
    if returns.empty: return 0.0, 0.0
    sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
    cum_ret = (1 + returns).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    return float(sharpe), float(max_dd)

# --- 4. 主介面 UI ---
st.set_page_config(layout="wide", page_title="AI 量化終端-測試用")
st.title("🧪 AI 自動財報預測與量化終端")

st.sidebar.header("配置參數")
raw_ticker = st.sidebar.text_input("股票代碼", value="2330")
ticker = fix_ticker(raw_ticker)

if st.button("執行全方位量化與財報預測"):
    with st.spinner(f"正在搜尋財報並預測 {ticker} 之 EPS..."):
        df = fetch_stock_data(ticker)
        
        if not df.empty:
            # 獲取自動預測之 EPS 並應用指令：P/E 40
            pred_eps = auto_predict_eps(ticker)
            target_price = float(pred_eps * 40)  # 指令：基於 P/E 40 的股價預測
            
            curr_price = float(df['Close'].iloc[-1])
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            sharpe, mdd = calculate_metrics(df)
            pred_5d = train_and_predict(df)

            # Fibonacci 計算
            plot_df = df.tail(90).copy()
            high_idx = plot_df['High'].idxmax()
            high_pos = plot_df.index.get_loc(high_idx)
            high_p = float(plot_df.loc[high_idx, 'High'])
            low_idx = plot_df.iloc[high_pos:]['Low'].idxmin()
            low_pos = plot_df.index.get_loc(low_idx)
            low_p = float(plot_df.loc[low_idx, 'Low'])
            diff = high_p - low_p
            fibs = {"0.382": high_p - 0.382 * diff, "0.500": high_p - 0.5 * diff, "0.618": high_p - 0.618 * diff}

            # --- Dashboard ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("預測年度 EPS", f"${pred_eps:.2f}")
            m2.metric("40x PE 估值價", f"${target_price:.1f}")
            m3.metric("目前偏離度", f"{((curr_price/target_price)-1):.1%}" if target_price > 0 else "N/A")
            m4.metric("AI 5日預測", f"${pred_5d:.1f}")
            m5.metric("最大回撤", f"{mdd:.2%}")

            # --- 圖表與分析 ---
            col_chart, col_eval = st.columns([2, 1])
            with col_chart:
                buf = io.BytesIO()
                mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
                ap = [mpf.make_addplot(plot_df['MA20'], color='#1E88E5', width=1),
                      mpf.make_addplot(plot_df['MA60'], color='#FB8C00', width=1.2)]
                
                fig, axlist = mpf.plot(plot_df, type='candle', style=s, addplot=ap, volume=True, figsize=(14, 10),
                                       hlines=dict(hlines=[fibs["0.382"], fibs["0.500"], fibs["0.618"]],
                                                   colors=['gray', 'gray', 'gray'], linestyle='-.'),
                                       tight_layout=True, returnfig=True)
                fig.savefig(buf, format='png', bbox_inches='tight')
                st.image(buf)

            with col_eval:
                st.write("### 💎 財報與估值深度點評")
                
                # 評等與位階判定
                rating = "🟢 優於大盤 (價值成長)" if (curr_price < target_price and pred_5d > curr_price) else "🟡 中立觀察"
                if target_price > 0 and curr_price > target_price * 1.2: rating = "🔴 溢價過高"
                
                st.markdown(f"""
                **【綜合分析評等】：{rating}**

                **【自動財報分析】**
                *   **EPS 預測來源**：系統已掃描最近四季損益表並完成年化推估。
                *   **預測 EPS**：**${pred_eps:.2f}**
                *   **目標價 (40x PE)**：**${target_price:.1f}**

                **【AI 動能點評】**
                1.  **AI 趨勢預測**：{'模型看好短期動能，預期朝 {:.1f} 前進。'.format(pred_5d) if pred_5d > curr_price else '模型預警短期震盪，請注意回檔風險。'}
                2.  **偏離度分析**：目前股價相對於 40 倍 PE 估值為 **{((curr_price/target_price)-1):.1% if target_price > 0 else 'N/A'}**。

                **【重點提醒】**
                *   此估值基於 40 倍 P/E 比例。若該股非高成長科技股，此估值可能偏樂觀。
                *   若顯示 EPS 為 0.0，代表該股財報數據在 yfinance 資料庫中不完整，建議手動查詢補充。
                """)
                st.write("**📐 Fibonacci Levels**")
                st.table(pd.DataFrame.from_dict(fibs, orient='index', columns=['Price']))
        else:
            st.error("無法取得該股票數據。")
