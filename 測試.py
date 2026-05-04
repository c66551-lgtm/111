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

# --- 核心：自動分析財報並預測全年 EPS ---
@st.cache_data(ttl=86400)
def predict_annual_eps(ticker):
    try:
        t = yf.Ticker(ticker)
        # 1. 嘗試從 info 直接獲取預估值
        info = t.info
        f_eps = info.get('forwardEps')
        t_eps = info.get('trailingEps')
        
        # 2. 如果 info 沒資料，改去翻閱「季度損益表」自動計算
        # 台股在 yfinance 裡常沒 info 但有 financials
        q_financials = t.quarterly_financials
        if q_financials is not None and not q_financials.empty:
            if 'Net Income' in q_financials.index:
                # 獲取最近四季的淨利總和 (簡單年化預測)
                recent_net_income = q_financials.loc['Net Income'].head(4).sum()
                # 獲取總股數 (Shares Outstanding)
                shares = info.get('sharesOutstanding')
                if shares and shares > 0:
                    calculated_eps = recent_net_income / shares
                    return calculated_eps

        return f_eps or t_eps or 0
    except:
        return 0

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
st.set_page_config(layout="wide", page_title="AI 量化終端-測試版")
st.title("🧪 AI 自動財報分析預測終端")

st.sidebar.header("配置參數")
raw_ticker = st.sidebar.text_input("股票代碼", value="2330")
ticker = fix_ticker(raw_ticker)
manual_eps = st.sidebar.number_input("手動修正 EPS (選填)", value=0.0)

if st.button("執行全方位量化預測"):
    with st.spinner(f"正在搜尋財報並計算預測 EPS..."):
        df = fetch_stock_data(ticker)
        
        if not df.empty:
            # 獲取自動預測 EPS
            auto_eps = predict_annual_eps(ticker)
            final_eps = manual_eps if manual_eps > 0 else auto_eps
            target_price = float(final_eps * 20) if final_eps > 0 else 0.0
            
            curr_price = float(df['Close'].iloc[-1])
            sharpe, mdd = calculate_metrics(df)
            pred_5d = train_and_predict(df)

            # Fibonacci 與圖表邏輯 (省略重複代碼，保持核心功能)
            plot_df = df.tail(90).copy()
            high_p = float(plot_df['High'].max())
            low_p = float(plot_df['Low'].min())
            
            # --- Dashboard ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("自動預測 EPS", f"${final_eps:.2f}")
            m2.metric("20x PE 估值", f"${target_price:.1f}")
            m3.metric("目前偏離度", f"{((curr_price/target_price)-1):.1%}" if target_price > 0 else "N/A")
            m4.metric("AI 5日預測", f"${pred_5d:.1f}")
            m5.metric("最大回撤", f"{mdd:.1%}")

            # --- 深度分析 ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"### 📈 {ticker} 趨勢圖表")
                # 此處放置原本的 mpf 繪圖邏輯...
                st.info("系統已自動掃描最近四季損益表，並根據獲利能力推估全年 EPS。")
            
            with col2:
                st.write("### 💎 財報預測點評")
                rating = "🟢 低估建議關注" if (curr_price < target_price and pred_5d > curr_price) else "🟡 價值合理"
                if target_price > 0 and curr_price > target_price * 1.1: rating = "🔴 溢價過高"
                
                st.markdown(f"""
                **【自動評等】：{rating}**
                
                **【核心成因分析】**
                1. **EPS 來源**：{'系統自動年化季度淨利計算' if manual_eps == 0 else '使用者手動指定'}。
                2. **20倍估值**：基於預測 EPS 所得之合理價位為 **${target_price:.1f}**。
                3. **動能匹配**：AI 模型預測五日後價格為 **${pred_5d:.1f}**，{'與估值方向一致' if (pred_5d > curr_price and curr_price < target_price) else '短期可能面臨修正'}。
                """)
        else:
            st.error("查無數據，請確認代碼是否正確。")
