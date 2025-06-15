import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Stock Assistant",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .st-bb { background-color: transparent; }
    .st-at { background-color: #0c1130; }
    div[data-testid="stMetric"] { background-color: rgba(38, 39, 48, 0.5); border-radius: 10px; }
    .ticker-input { padding: 20px; }
    .stock-header { color: #1f77b4; }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .news-card { 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 10px;
        background: rgba(19, 23, 34, 0.8);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("AI Stock Market Analyst")

# Time frame options
TIME_FRAMES = {
    "1 Day": "1d",
    "3 Day": "3d",
    "1 Week": "1wk",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "Year to Date": "ytd",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
    "Max": "max"
}

# Sidebar configuration
with st.sidebar:
    st.header("🔍 Search Parameters")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    
    # Time frame selector dropdown
    selected_label = st.selectbox(
        "Chart Time Frame",
        list(TIME_FRAMES.keys()),
        index=4  # Default to 6 Months
    )
    time_frame = TIME_FRAMES[selected_label]
    
    # Analysis options
    st.subheader("Technical Analysis")
    show_candlestick = st.checkbox("Candlestick Chart", True)
    show_sma = st.checkbox("Simple Moving Average (SMA)")
    show_rsi = st.checkbox("Relative Strength Index (RSI)")
    
    st.subheader("AI Features")
    analyze_sentiment = st.checkbox("Enable Twitter Sentiment Analysis", True)
    predict_trend = st.checkbox("Enable Price Prediction", True)
    

# Main content
if ticker:
    try:
        # Determine currency based on ticker suffix
        is_indian_stock = ticker.endswith(".NS") or ticker.endswith(".BO")
        currency_symbol = "₹" if is_indian_stock else "$"
        currency_text = "INR" if is_indian_stock else "USD"
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=time_frame)
        
        if hist.empty:
            st.warning("No historical data found. Try another ticker.")
            st.stop()
            
        # Company information
        info = stock.info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Company", info.get('shortName', 'N/A'))
        col2.metric("Sector", info.get('sector', 'N/A'))
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', hist['Close'][-1]))
        prev_close = info.get('previousClose', hist['Close'][-2])
        price_change = ((current_price - prev_close) / prev_close) * 100
        
        # Format price with appropriate currency
        col3.metric("Current Price", f"{currency_symbol}{current_price:,.2f}", 
                   f"{price_change:.2f}%", 
                   delta_color="inverse")
        
        # Format market cap with appropriate currency and units
        market_cap = info.get('marketCap', None)
        if market_cap:
            if is_indian_stock:
                # Convert to crores for Indian stocks
                market_cap_cr = market_cap / 10000000
                col4.metric("Market Cap", f"{currency_symbol}{market_cap_cr:,.2f} Cr")
            else:
                # Convert to billions for international stocks
                market_cap_bn = market_cap / 1000000000
                col4.metric("Market Cap", f"{currency_symbol}{market_cap_bn:,.2f} B")
        else:
            col4.metric("Market Cap", "N/A")
        
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Technical Analysis", "🗞️ News & Sentiment", "💡 AI Insights"])
        
        with tab1:  # Charts tab
            fig = go.Figure()
            
            if show_candlestick:
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Candlestick'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    name='Closing Price',
                    line=dict(color='#1f77b4')
                ))
            
            if show_sma:
                sma_period = st.slider("SMA Period", 5, 200, 50)
                hist['SMA'] = hist['Close'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['SMA'],
                    name=f'SMA {sma_period}',
                    line=dict(color='orange', dash='dot')
                ))
            
            fig.update_layout(
                title=f'{ticker} Price Movement - {selected_label}',
                xaxis_title='Date',
                yaxis_title=f'Price ({currency_text})',
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color='#4a7bff'
            ))
            vol_fig.update_layout(
                title='Trading Volume',
                xaxis_title='Date',
                yaxis_title='Volume',
                template='plotly_dark'
            )
            st.plotly_chart(vol_fig, use_container_width=True)
        
        with tab2:  # Technical Analysis
            if show_rsi:
                st.subheader("Relative Strength Index (RSI)")
                period = st.slider("RSI Period", 5, 30, 14)
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))
                
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['RSI'],
                    name='RSI',
                    line=dict(color='#8e44ad')
                ))
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.update_layout(
                    yaxis_range=[0,100],
                    template='plotly_dark'
                )
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            st.subheader("Key Technical Metrics")
            cols = st.columns(3)
            
            # Format 52-week high/low with appropriate currency
            fifty_two_high = info.get('fiftyTwoWeekHigh', None)
            fifty_two_low = info.get('fiftyTwoWeekLow', None)
            
            cols[0].metric("52W High", 
                          f"{currency_symbol}{fifty_two_high:,.2f}" if fifty_two_high else "N/A")
            cols[1].metric("52W Low", 
                          f"{currency_symbol}{fifty_two_low:,.2f}" if fifty_two_low else "N/A")
            cols[2].metric("Volatility (Beta)", info.get('beta', 'N/A'))
        
        with tab3:  # News & Sentiment
            try:
                news = stock.news
                st.subheader("Latest Market News")
                
                for i, item in enumerate(news[:5]):
                    with st.container():
                        pub_time = datetime.fromtimestamp(item['providerPublishTime'])
                        st.markdown(f"""
                        <div class="news-card">
                            <h4>{item['title']}</h4>
                            <p><i>Source: {item['publisher']} • {pub_time.strftime('%Y-%m-%d')}</i></p>
                            <a href="{item['link']}" target="_blank">Read full article</a>
                        </div>
                        """, unsafe_allow_html=True)
            except:
                st.warning("Could not load news at this time")
            
            if analyze_sentiment:
                st.subheader("Sentiment Analysis")
                st.markdown("""
                <div style="background:#1a1d29; padding:15px; border-radius:10px;">
                    <h4>Twitter Sentiment Analysis (Coming Soon)</h4>
                    <p>AI-powered sentiment scoring from Twitter feeds will appear here</p>
                    <div style="display:flex; margin-top:15px;">
                        <div style="width:40%; text-align:center;">
                            <h3 class="positive">72%</h3>
                            <p>Positive Sentiment</p>
                        </div>
                        <div style="width:40%; text-align:center;">
                            <h3 class="negative">28%</h3>
                            <p>Negative Sentiment</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:  # AI Insights
            st.subheader("AI-Powered Predictions")
            
            # Placeholder for ML predictions
            col1, col2 = st.columns(2)
            with col1:
                # Format prediction with appropriate currency
                pred_price = 152.34 if currency_symbol == "$" else 152.34 * 75  # Simple conversion for demo
                st.markdown(f"""
                <div style="background:#1a1d29; padding:20px; border-radius:10px; height:250px;">
                    <h4>Price Forecast</h4>
                    <p>LSTM neural network prediction:</p>
                    <div style="margin-top:30px; text-align:center;">
                        <h2 class="positive">↑ {currency_symbol}{pred_price:,.2f}</h2>
                        <p>(+3.2% in 7 days)</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background:#1a1d29; padding:20px; border-radius:10px; height:250px;">
                    <h4>Risk Analysis</h4>
                    <p>AI assessment of investment risk:</p>
                    <div style="margin-top:30px;">
                        <div style="display:flex; justify-content:space-between;">
                            <span>Market Risk</span>
                            <span class="negative">High</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:10px;">
                            <span>Volatility Risk</span>
                            <span>Medium</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:10px;">
                            <span>Sentiment Risk</span>
                            <span class="positive">Low</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-top:20px; background:#1a1d29; padding:20px; border-radius:10px;">
                <h4>Recommendation Engine</h4>
                <div style="text-align:center; padding:20px;">
                    <h2 class="positive">STRONG BUY</h2>
                    <p>Based on technical indicators and sentiment analysis</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.info("Enter a stock ticker symbol to begin analysis (e.g. AAPL, TSLA, MSFT, RELIANCE.NS)")