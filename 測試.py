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

@st.cache_data(ttl=86400)
def get_ticker_info(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info
    except:
        return {}

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

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror",
                         random_state=42)
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
st.title("🧪 AI 量化研究終端 (測試版)")

# --- 側邊欄設定 ---
st.sidebar.header("配置參數")
raw_ticker = st.sidebar.text_input("股票代碼", value="2330")
ticker = fix_ticker(raw_ticker)

# 修正重點：新增手動輸入 EPS 的功能
manual_eps = st.sidebar.number_input("手動輸入預估 EPS (若估值顯示 N/A)", value=0.0, step=0.1)

if st.button("執行全方位量化分析"):
    with st.spinner(f"正在連線全球數據中心，進行深度點評 {ticker} ..."):
        df = fetch_stock_data(ticker)

        if not df.empty:
            curr_price = float(df['Close'].iloc[-1])
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            sharpe, mdd = calculate_metrics(df)
            pred_5d = train_and_predict(df)

            # Fibonacci 與波段計算
            plot_df = df.tail(90).copy()
            high_idx = plot_df['High'].idxmax()
            high_pos = plot_df.index.get_loc(high_idx)
            high_p = float(plot_df.loc[high_idx, 'High'])

            after_high_df = plot_df.iloc[high_pos:]
            low_idx = after_high_df['Low'].idxmin()
            low_pos = plot_df.index.get_loc(low_idx)
            low_p = float(plot_df.loc[low_idx, 'Low'])

            diff = high_p - low_p
            fibs = {"0.382": high_p - 0.382 * diff, "0.500": high_p - 0.5 * diff, "0.618": high_p - 0.618 * diff}

            # --- 修正後的估值計算邏輯 ---
            info = get_ticker_info(ticker)
            # 優先使用手動輸入的 EPS，若無則嘗試從 API 抓取
            auto_eps = manual_eps if manual_eps > 0 else (info.get('forwardEps') or info.get('trailingEps') or 0)
            target_price = float(auto_eps * 20) if auto_eps else 0.0

            # --- Dashboard ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("趨勢位階", "🚀 多頭" if df['MA20'].iloc[-1] > df['MA60'].iloc[-1] else "📉 空頭")
            m2.metric("夏普值 (Sharpe)", f"{sharpe:.2f}")
            m3.metric("最大回撤", f"{mdd:.2%}")
            m4.metric("AI 5日預測", f"${pred_5d:.1f}")
            m5.metric("20x PE 估值", f"${target_price:.1f}" if target_price > 0 else "N/A")

            # --- 圖表與深度分析區 ---
            col_chart, col_eval = st.columns([2, 1])

            with col_chart:
                buf = io.BytesIO()
                mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
                ap = [mpf.make_addplot(plot_df['MA20'], color='#1E88E5', width=1),
                      mpf.make_addplot(plot_df['MA60'], color='#FB8C00', width=1.2)]

                y_min, y_max = plot_df['Low'].min() * 0.98, high_p * 1.03
                fig, axlist = mpf.plot(plot_df, type='candle', style=s, addplot=ap, volume=True, figsize=(14, 10),
                                       hlines=dict(hlines=[fibs["0.382"], fibs["0.500"], fibs["0.618"]],
                                                   colors=['gray', 'gray', 'gray'], linestyle='-.'),
                                       tight_layout=True, returnfig=True, ylim=(y_min, y_max))

                ax = axlist[0]
                ax.annotate(f'High: {high_p:.1f}', xy=(high_pos, high_p), xytext=(0, 10), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'), ha='center', color='red', fontweight='bold')
                ax.annotate(f'Low: {low_p:.1f}', xy=(low_pos, low_p), xytext=(0, -30), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='green'), ha='center', color='green', fontweight='bold')

                fig.savefig(buf, format='png', bbox_inches='tight')
                st.image(buf)

            with col_eval:
                st.write("### 💎 深度點評")
                rating = "🟢 優於大盤" if (pred_5d > curr_price and (target_price == 0 or curr_price < target_price)) else "🟡 中立持股"
                if target_price > 0 and curr_price > target_price and pred_5d < curr_price: rating = "🔴 減碼迴備"
                
                wave_stage = "主升段 / 強勢整理"
                if curr_price < low_p: wave_stage = "C波延伸 (偏空)"
                elif curr_price < fibs["0.500"]: wave_stage = "C波打底階段"
                elif curr_price < fibs["0.382"]: wave_stage = "B波反彈末端"

                dev_text = f"{((curr_price / target_price) - 1):.1%}" if target_price > 0 else "N/A"

                st.markdown(f"""
                **【綜合分析評等】：{rating}**

                **【波浪位階判定】**
                目前處於：**{wave_stage}**

                **【關鍵位階與估值共振】**
                *   **估值分析 (20x PE)**：
                    目前合理估值為 **${target_price:.1f}**。
                    偏離度為 **{dev_text}**。
                
                *(註：若顯示 N/A，請於左側手動輸入該股 EPS)*
                """)
                st.table(pd.DataFrame.from_dict(fibs, orient='index', columns=['Price']))
        else:
            st.error("無法取得該股票數據。")
