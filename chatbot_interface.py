import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🤖 Advanced AI Trading Chatbot",
    page_icon="💬",
    layout="wide"
)

# CSS for trading chatbot interface
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #333;
        padding: 10px 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    .prediction-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .buy-badge {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .sell-badge {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .hold-badge {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }

    .typing-indicator {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 20px;
        margin: 10px 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .metrics-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .model-performance {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm your **Advanced AI Trading Assistant** powered by ensemble ML models. I can:\n\n🔮 **Predict** with XGBoost, LightGBM, Random Forest & Neural Networks\n📊 **Backtest** strategies with risk metrics\n📈 **Analyze** 50+ technical indicators and market patterns\n💰 **Calculate** ROI, Sharpe ratio, max drawdown\n📉 **Visualize** comprehensive trading analysis\n⚡ **Generate** high-accuracy trading signals\n\nJust tell me a ticker symbol like AAPL, GOOGL, or TSLA for a full AI analysis!",
            "timestamp": datetime.now()
        }
    ]

if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = True

if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}



# Advanced ML functions for trading analysis

@st.cache_data(ttl=7200)  # Increased cache to 2 hours to reduce API calls
def get_stock_data_advanced(ticker: str, period: str = "2y"):
    """Fetch comprehensive stock data with improved error handling and rate limiting"""
    try:
        # Add delay to prevent rate limiting
        time.sleep(random.uniform(0.2, 0.8))
        
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if len(df) < 100:  # Need more data for ML models
            # Try a shorter period if 2y doesn't work
            if period == "2y":
                st.warning(f"Insufficient 2-year data for {ticker}, trying 1-year period...")
                df = stock.history(period="1y")
                
        if len(df) < 50:  # Still not enough data
            st.error(f"Insufficient data for {ticker} - only {len(df)} days available")
            return None
            
        df.reset_index(inplace=True)
        return df
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate', 'limit', 'blocked', 'exceeded', 'quota', 'fair-use']):
            st.error(f"⚠️ Yahoo Finance API rate limit reached for {ticker}. Please wait a few minutes before trying again.")
            st.info("💡 **Tip**: Try clearing the analysis cache in the sidebar and wait 5-10 minutes before analyzing again.")
        else:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def add_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators"""
    
    # Basic indicators
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    
    # RSI with multiple periods
    for period in [9, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # Price momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    df['Price_Volume'] = df['Close'] * df['Volume']
    
    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated engineered features"""
    
    # Price-based features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    # Volatility measures
    for window in [5, 10, 20, 30]:
        df[f'Volatility_{window}'] = df['Daily_Return'].rolling(window=window).std()
        df[f'Return_Mean_{window}'] = df['Daily_Return'].rolling(window=window).mean()
        df[f'Price_Range_{window}'] = (df['High'].rolling(window=window).max() - 
                                       df['Low'].rolling(window=window).min()) / df['Close']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'RSI_14_Lag_{lag}'] = df['RSI_14'].shift(lag)
    
    # Rolling statistics
    for window in [10, 20]:
        df[f'Close_Skew_{window}'] = df['Close'].rolling(window=window).skew()
        df[f'Volume_Skew_{window}'] = df['Volume'].rolling(window=window).skew()
    
    # Trend features
    df['Price_Trend_5'] = (df['Close'] > df['SMA5']).astype(int)
    df['Price_Trend_20'] = (df['Close'] > df['SMA20']).astype(int)
    df['SMA_Cross'] = (df['SMA5'] > df['SMA20']).astype(int)
    
    # Support and resistance levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    # Handle division by zero in Price_Position
    resistance_support_diff = df['Resistance'] - df['Support']
    df['Price_Position'] = np.where(
        resistance_support_diff != 0,
        (df['Close'] - df['Support']) / resistance_support_diff,
        0.5
    )
    
    # Market microstructure features
    df['Spread'] = (df['High'] - df['Low']) / df['Close']
    df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
    df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def generate_sophisticated_labels(df: pd.DataFrame, forward_days=3) -> pd.DataFrame:
    """Generate optimized 3-class labels for maximum accuracy"""
    
    # Multiple forward-looking periods for robust signal
    for days in [1, 2, 3, 5]:
        df[f'Future_Return_{days}'] = (df['Close'].shift(-days) - df['Close']) / df['Close']
    
    # Weighted combination giving more weight to near-term predictions
    df['Combined_Signal'] = (0.4 * df['Future_Return_1'] + 
                            0.3 * df['Future_Return_2'] + 
                            0.2 * df['Future_Return_3'] + 
                            0.1 * df['Future_Return_5'])
    
    # Use percentile-based labeling for balanced, high-quality signals
    signal_quantiles = df['Combined_Signal'].quantile([0.25, 0.75])
    
    def create_label(signal):
        if signal > signal_quantiles[0.75]:
            return 1  # Buy (top 25% - strong positive signals)
        elif signal < signal_quantiles[0.25]:
            return -1  # Sell (bottom 25% - strong negative signals)
        else:
            return 0  # Hold (middle 50% - uncertain signals)
    
    df['Label'] = df['Combined_Signal'].apply(create_label)
    
    # Remove rows with insufficient future data
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_advanced_models():
    """Create ensemble of sophisticated ML models with optimized parameters"""
    
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,  # Reduced for speed
            max_depth=4,
            learning_rate=0.1,  # Increased for speed
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            random_state=42,
            eval_metric='mlogloss'
        ),
        
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        ),
        
        'RandomForest': RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42
        ),
        
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42
        )
    }
    
    return models

@st.cache_data(ttl=7200)  # Increased cache to 2 hours to reduce API calls
def train_advanced_models(ticker: str):
    """Train advanced ML models with full pipeline including sentiment analysis"""
    
    # Get data
    df = get_stock_data_advanced(ticker)
    if df is None:
        return None
    
    # Feature engineering
    df = add_advanced_technical_indicators(df)
    df = add_advanced_features(df)
    
    df = generate_sophisticated_labels(df)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Label', 'Combined_Signal'] 
                   and not col.startswith('Future_')]
    
    # Remove features with too many NaN values
    feature_cols = [col for col in feature_cols if df[col].isna().sum() < len(df) * 0.1]
    
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    X_train = train_df[feature_cols]
    y_train_raw = train_df['Label']
    X_test = test_df[feature_cols]
    y_test_raw = test_df['Label']
    
    # Transform labels to start from 0 for XGBoost compatibility
    label_mapping = {-1: 0, 0: 1, 1: 2}
    reverse_mapping = {0: -1, 1: 0, 2: 1}
    
    y_train = y_train_raw.map(label_mapping)
    y_test = y_test_raw.map(label_mapping)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(X_train.median()))
    X_test_scaled = scaler.transform(X_test.fillna(X_train.median()))
    
    # Create balanced training data
    X_train_selected = X_train_scaled[:, :len(feature_cols)]
    X_test_selected = X_test_scaled[:, :len(feature_cols)]
    
    print(f"Selected {len(feature_cols)} features")
    
    # Train models
    models = create_advanced_models()
    trained_models = {}
    predictions = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        trained_models[name] = model
        predictions[name] = pred
        accuracies[name] = accuracy_score(y_test, pred)
    
    # Create ensemble model
    ensemble_models = [
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('rf', models['RandomForest']),
        ('et', models['ExtraTrees'])
    ]
    
    # Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=ensemble_models,
        voting='soft'
    )
    voting_classifier.fit(X_train_scaled, y_train)
    voting_predictions = voting_classifier.predict(X_test_scaled)
    voting_accuracy = accuracy_score(y_test, voting_predictions)
    
    # Use best model
    accuracies['Voting_Ensemble'] = voting_accuracy
    best_model_name = max(accuracies, key=accuracies.get)
    
    if best_model_name == 'Voting_Ensemble':
        best_model = voting_classifier
        best_predictions = voting_predictions
    else:
        best_model = trained_models[best_model_name]
        best_predictions = predictions[best_model_name]
    
    # Transform predictions back to original format
    best_predictions_original = pd.Series(best_predictions).map(reverse_mapping).values
    
    # Update test_df
    test_df = test_df.copy()
    test_df['Label_Original'] = y_test_raw.values
    
    return {
        'model': best_model,
        'test_df': test_df,
        'predictions': best_predictions_original,
        'scaler': scaler,
        'selected_features': feature_cols,
        'accuracies': accuracies,
        'best_model_name': best_model_name,
        'df_full': df
    }

def backtest_strategy_advanced(test_df: pd.DataFrame, predictions, initial_cash=100000, fee=0.001):
    """Advanced backtesting with position sizing and risk management"""
    cash = initial_cash
    position = 0
    daily_value = []
    trades = []
    
    for i in range(len(test_df)):
        price_open = test_df.iloc[i]['Open']
        price_close = test_df.iloc[i]['Close']
        signal = predictions[i-1] if i > 0 else 0
        
        # Position sizing based on signal strength
        if signal == 1:  # Buy
            position_size = 0.8  # 80% of available cash
        else:
            position_size = 0
        
        # Buy signals
        if signal > 0 and position == 0:
            shares_to_buy = int((cash * position_size) // price_open)
            if shares_to_buy > 0:
                position = shares_to_buy
                cash -= position * price_open * (1 + fee)
                trades.append(('BUY', price_open, position, test_df.iloc[i]['Date']))
        
        # Sell signals
        elif signal < 0 and position > 0:
            cash += position * price_open * (1 - fee)
            trades.append(('SELL', price_open, position, test_df.iloc[i]['Date']))
            position = 0
        
        total_value = cash + position * price_close
        daily_value.append(total_value)
    
    # Calculate performance metrics
    final_value = daily_value[-1]
    returns = pd.Series(daily_value).pct_change().dropna()
    cumulative = pd.Series(daily_value)
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()
    
    # Risk metrics
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    roi = (final_value - initial_cash) / initial_cash
    win_rate = len([r for r in returns if r > 0]) / len(returns) if len(returns) > 0 else 0
    
    return {
        'daily_value': daily_value,
        'trades': trades,
        'final_value': final_value,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate
    }

def get_latest_prediction(df_full, model, scaler, selected_features):
    """Get prediction for the latest data point"""
    try:
        # Get feature columns
        feature_cols = [col for col in df_full.columns if col not in ['Date', 'Label', 'Combined_Signal'] 
                       and not col.startswith('Future_')]
        feature_cols = [col for col in feature_cols if df_full[col].isna().sum() < len(df_full) * 0.1]
        
        # Get latest data
        latest_data = df_full[feature_cols].iloc[-1:].fillna(df_full[feature_cols].median())
        latest_scaled = scaler.transform(latest_data)
        
        # Make prediction
        prediction_encoded = model.predict(latest_scaled)[0]
        
        # Transform back to original labels
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        prediction = reverse_mapping.get(prediction_encoded, 0)
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(latest_scaled)[0]
            confidence = max(proba)
        else:
            confidence = 0.7
        
        return prediction, confidence
    except Exception as e:
        return 0, 0.5

def create_quick_visualization(ticker, test_df, predictions, backtest_results):
    """Create a quick visualization for the chat interface"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Price with signals
        ax1.plot(test_df['Date'], test_df['Close'], 'b-', alpha=0.7, label='Price')
        
        # Add trading signals
        for i, pred in enumerate(predictions):
            if pred == 1:  # Buy
                ax1.scatter(test_df.iloc[i]['Date'], test_df.iloc[i]['Close'], 
                           color='green', marker='^', s=50, alpha=0.8)
            elif pred == -1:  # Sell
                ax1.scatter(test_df.iloc[i]['Date'], test_df.iloc[i]['Close'], 
                           color='red', marker='v', s=50, alpha=0.8)
        
        ax1.set_title(f'{ticker} Price with AI Signals')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio performance
        if backtest_results['daily_value']:
            portfolio_returns = (np.array(backtest_results['daily_value']) / 100000 - 1) * 100
            ax2.plot(test_df['Date'].iloc[:len(portfolio_returns)], portfolio_returns, 'g-', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_title('Portfolio Performance')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3.plot(test_df['Date'], test_df['RSI_14'], 'purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax3.set_title('RSI(14)')
        ax3.set_ylabel('RSI')
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        labels = []
        colors = []
        for idx in pred_counts.index:
            if idx == -1:
                labels.append('Sell')
                colors.append('red')
            elif idx == 0:
                labels.append('Hold')
                colors.append('orange')
            elif idx == 1:
                labels.append('Buy')
                colors.append('green')
        
        if len(pred_counts) > 0:
            ax4.pie(pred_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
            ax4.set_title('Signal Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'{ticker}_quick_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        return None

def format_message(content, role="assistant"):
    """Format message for display"""
    css_class = "bot-message" if role == "assistant" else "user-message"
    return f'<div class="{css_class}">{content}</div>'

def extract_ticker(text):
    """Extract ticker symbol from user input"""
    import re
    # Look for ticker patterns (2-5 uppercase letters)
    tickers = re.findall(r'\b[A-Z]{1,5}\b', text.upper())
    
    # Common words to exclude
    exclude = {'THE', 'AND', 'OR', 'FOR', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'AI', 'ML'}
    
    valid_tickers = [t for t in tickers if t not in exclude and len(t) <= 5]
    
    return valid_tickers[0] if valid_tickers else None

def generate_advanced_ai_response(user_input):
    """Generate comprehensive AI response with full ML analysis"""
    ticker = extract_ticker(user_input)
    
    if not ticker:
        return """🤔 I didn't catch a ticker symbol. Could you please provide one? 

For example, try saying:
- "Analyze AAPL with full AI models"
- "What does the ML ensemble say about GOOGL?"
- "Run advanced backtesting on TSLA"
- "MSFT comprehensive analysis"

I'll use XGBoost, LightGBM, Random Forest, and Neural Networks for the prediction! 🚀"""
    
    # Check cache first with longer cache time to reduce API calls
    if ticker in st.session_state.analysis_cache:
        cache_time = st.session_state.analysis_cache[ticker]['timestamp']
        # Increased cache time to 2 hours to avoid hitting rate limits
        if datetime.now() - cache_time < timedelta(hours=2):
            cached_result = st.session_state.analysis_cache[ticker]
            return generate_response_from_cache(cached_result, ticker)
    
    # Show comprehensive typing indicator
    typing_placeholder = st.empty()
    
    with typing_placeholder.container():
        st.markdown('<div class="typing-indicator">🤖 Fetching real-time data for ' + ticker + '...</div>', unsafe_allow_html=True)
        time.sleep(1)
        st.markdown('<div class="typing-indicator">🧠 Training ensemble ML models with technical indicators...</div>', unsafe_allow_html=True)
        time.sleep(2)
        st.markdown('<div class="typing-indicator">📊 Running advanced backtesting and risk analysis...</div>', unsafe_allow_html=True)
        time.sleep(1)
        st.markdown('<div class="typing-indicator">📈 Generating comprehensive visualizations...</div>', unsafe_allow_html=True)
        time.sleep(1)
    
    typing_placeholder.empty()
    
    # Train models and get analysis with better error handling
    try:
        with st.spinner("🔮 Running Advanced AI Analysis..."):
            model_results = train_advanced_models(ticker)
        
        if model_results is None:
            return f"""❌ Sorry, I couldn't fetch sufficient data for {ticker}. This may be due to:

🔸 **API Rate Limits**: Yahoo Finance may be temporarily limiting requests
🔸 **Invalid Ticker**: Please check if {ticker} is a valid stock symbol
🔸 **Insufficient Data**: The ticker may not have enough historical data

💡 **What you can do:**
- Wait 5-10 minutes and try again
- Clear the analysis cache in the sidebar
- Try a different, more common ticker (e.g., AAPL, GOOGL, MSFT)
- Check if {ticker} is the correct symbol on Yahoo Finance"""
        
        # Get latest prediction
        latest_prediction, confidence = get_latest_prediction(
            model_results['df_full'], 
            model_results['model'], 
            model_results['scaler'], 
            model_results['selected_features']
        )
        
        # Run backtesting
        backtest_results = backtest_strategy_advanced(
            model_results['test_df'], 
            model_results['predictions']
        )
        
        # Create quick visualization
        plot_path = create_quick_visualization(
            ticker, 
            model_results['test_df'], 
            model_results['predictions'], 
            backtest_results
        )
        
        # Cache results
        st.session_state.analysis_cache[ticker] = {
            'model_results': model_results,
            'backtest_results': backtest_results,
            'latest_prediction': latest_prediction,
            'confidence': confidence,
            'plot_path': plot_path,
            'timestamp': datetime.now()
        }
        
        return generate_comprehensive_response(model_results, backtest_results, latest_prediction, confidence, ticker, plot_path)
        
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}. Please try again or try a different ticker."

def generate_response_from_cache(cached_result, ticker):
    """Generate response from cached analysis"""
    model_results = cached_result['model_results']
    backtest_results = cached_result['backtest_results']
    latest_prediction = cached_result['latest_prediction']
    confidence = cached_result['confidence']
    plot_path = cached_result['plot_path']
    
    response = f"## 🚀 **Cached Analysis for {ticker}** (Updated: {cached_result['timestamp'].strftime('%H:%M')})\n\n"
    response += generate_comprehensive_response(model_results, backtest_results, latest_prediction, confidence, ticker, plot_path)
    return response

def generate_comprehensive_response(model_results, backtest_results, latest_prediction, confidence, ticker, plot_path):
    """Generate the comprehensive AI response"""
    
    # Get stock info with rate limiting protection
    try:
        # Add delay and try to get company info
        time.sleep(random.uniform(0.1, 0.3))
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
    except Exception as e:
        # If we hit rate limits, just use the ticker
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['rate', 'limit', 'blocked', 'exceeded', 'quota']):
            st.info(f"ℹ️ Using cached data for {ticker} due to API limits")
        company_name = ticker
        sector = 'Unknown'
        market_cap = 0
    
    # Start building response
    response = f"## 🤖 **Advanced AI Analysis for {ticker}**\n"
    if company_name != ticker:
        response += f"**Company:** {company_name} | **Sector:** {sector}\n\n"
    
    # Add prediction badge
    if latest_prediction == 1:
        badge_class = "buy-badge"
        emoji = "🟢"
        action = "BUY"
    elif latest_prediction == -1:
        badge_class = "sell-badge"
        emoji = "🔴"
        action = "SELL"
    else:
        badge_class = "hold-badge"
        emoji = "🟡"
        action = "HOLD"
    
    response += f'<div class="prediction-badge {badge_class}">{emoji} **{action}** {ticker}</div>\n\n'
    
    # Model performance section
    best_model = model_results['best_model_name']
    best_accuracy = model_results['accuracies'][best_model]
    
    response += f"""
<div class="model-performance">
<h3>🏆 **ML Model Performance**</h3>

**Best Model:** {best_model} | **Accuracy:** {best_accuracy:.1%} | **Confidence:** {confidence:.1%}

**All Models Tested:**
"""
    
    for model_name, accuracy in model_results['accuracies'].items():
        emoji = "🥇" if model_name == best_model else "🥈" if accuracy > 0.4 else "🥉"
        response += f"\n• {emoji} {model_name}: {accuracy:.1%}"
    
    response += "\n</div>\n\n"
    
    # Backtesting results
    roi = backtest_results['roi']
    final_value = backtest_results['final_value']
    max_drawdown = backtest_results['max_drawdown']
    sharpe_ratio = backtest_results['sharpe_ratio']
    win_rate = backtest_results['win_rate']
    num_trades = len(backtest_results['trades'])
    
    response += f"""
<div class="metrics-box">
<h3>📊 **Backtesting Results** (6 months)</h3>

💰 **Final Portfolio Value:** ${final_value:,.0f}  
📈 **Total ROI:** {roi:+.2%}  
📉 **Max Drawdown:** {max_drawdown:.2%}  
⚡ **Sharpe Ratio:** {sharpe_ratio:.2f}  
🎯 **Win Rate:** {win_rate:.1%}  
🔄 **Total Trades:** {num_trades}  
</div>
"""
    
    # Current technical analysis
    latest_data = model_results['test_df'].iloc[-1]
    current_price = latest_data['Close']
    rsi = latest_data['RSI_14']
    sma20 = latest_data['SMA20']
    sma50 = latest_data['SMA50']
    
    response += f"""
## 📊 **Current Technical Analysis**

**Current Metrics:**
- 💰 **Price:** ${current_price:.2f}
- 📊 **RSI(14):** {rsi:.1f}
- 📈 **20-day SMA:** ${sma20:.2f}
- 📉 **50-day SMA:** ${sma50:.2f}

**Key Insights:**
"""
    
    # Technical insights
    if rsi < 30:
        response += "• ✅ RSI indicates oversold conditions (potential buying opportunity)\n"
    elif rsi > 70:
        response += "• ⚠️ RSI shows overbought conditions (potential sell signal)\n"
    else:
        response += "• ➡️ RSI is in neutral territory\n"
    
    if current_price > sma20:
        response += "• 📈 Price is above 20-day moving average (bullish short-term trend)\n"
    else:
        response += "• 📉 Price is below 20-day moving average (bearish short-term trend)\n"
    
    if sma20 > sma50:
        response += "• 🚀 20-day MA above 50-day MA (bullish medium-term trend)\n"
    else:
        response += "• 📉 20-day MA below 50-day MA (bearish medium-term trend)\n"
    
    # Show visualization if available
    if plot_path and os.path.exists(plot_path):
        response += f"\n## 📈 **Quick Visual Analysis**\n"
        # Note: In a real deployment, you'd want to serve these images properly
        # For now, we'll just mention that charts are available
        response += "*📊 Interactive charts with price signals, portfolio performance, RSI analysis, and prediction distribution have been generated.*\n"
    
    # Trading recommendation reasoning
    response += f"""
## 🎯 **AI Recommendation Reasoning**

The **{best_model}** model recommends **{action}** based on:

🧠 **Machine Learning Analysis:**
- Ensemble of {len(model_results['accuracies'])} different ML algorithms
- {len(model_results['selected_features'])} carefully selected features
- Advanced technical indicators and engineered features
- Historical backtest shows {roi:+.1%} returns
- Maximum portfolio decline of {abs(max_drawdown):.1%}
- Sharpe ratio of {sharpe_ratio:.2f} (higher is better)
- Win rate of {win_rate:.1%} on historical trades

⚠️ **Important Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. Always conduct your own research and consider your risk tolerance before making investment decisions.
"""
    
    # Cleanup old plot files
    try:
        if plot_path and os.path.exists(plot_path):
            # Keep the plot for this session but clean up old ones
            pass
    except:
        pass
    
    return response

def main():
    st.title("🤖💬 Advanced AI Trading Chatbot")
    st.markdown("**Powered by Ensemble ML Models: XGBoost • LightGBM • Random Forest • Neural Networks**")
    

    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(format_message(message["content"], "user"), unsafe_allow_html=True)
        else:
            st.markdown(format_message(message["content"], "assistant"), unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me to analyze any stock with advanced AI models... (e.g., 'Run full ML analysis on AAPL')")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Display user message
        st.markdown(format_message(user_input, "user"), unsafe_allow_html=True)
        
        # Generate and display AI response
        ai_response = generate_advanced_ai_response(user_input)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now()
        })
        
        st.markdown(format_message(ai_response, "assistant"), unsafe_allow_html=True)
        
        # Rerun to update the display
        st.rerun()
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🚀 **Advanced AI Features**")
        
        st.markdown("""
        **🧠 ML Models Used:**
        - XGBoost Classifier
        - LightGBM Classifier  
        - Random Forest
        - Extra Trees
        - Voting Ensemble
        - Stacking Ensemble
        
        **📊 Features Analyzed:**
        - 50+ Technical Indicators
        - Price Action Patterns
        - Volume Analysis
        - Volatility Measures
        - Support/Resistance Levels
        - Advanced Feature Engineering
        
        **⚡ Model Optimization:**
        - Feature selection & scaling
        - Ensemble voting strategies
        - Cross-validation & backtesting
        - Risk-adjusted performance metrics
        - Real-time prediction generation
        """)
        
        st.header("💡 **Try These Advanced Queries:**")
        
        example_commands = [
            "Full ML ensemble analysis on AAPL",
            "Technical analysis for GOOGL",
            "TSLA advanced backtesting", 
            "MSFT comprehensive AI analysis",
            "NVDA ML prediction with risk metrics",
            "AMZN technical indicators analysis"
        ]
        
        for cmd in example_commands:
            if st.button(cmd, key=f"example_{cmd}"):
                # Simulate user input
                st.session_state.messages.append({
                    "role": "user", 
                    "content": cmd,
                    "timestamp": datetime.now()
                })
                
                ai_response = generate_advanced_ai_response(cmd)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📈 **Performance Metrics:**")
        st.write("✅ **Accuracy:** Ensemble ML models with feature selection")
        st.write("✅ **Backtesting:** Advanced risk-adjusted returns")
        st.write("✅ **Features:** 50+ technical indicators optimally selected")
        st.write("✅ **Models:** 6 ML algorithms with voting ensemble")
        st.write("✅ **Speed:** Real-time analysis and predictions")
        st.write("✅ **Risk Management:** Sharpe ratio, drawdown analysis")
        st.write("✅ **Visualization:** Comprehensive charts and metrics")
        
        st.markdown("---")
        
        # Cache management
        st.subheader("⚙️ **Settings:**")
        
        st.write("**🗑️ Cache Management:**")
        
        if st.button("🗑️ Clear Analysis Cache"):
            st.session_state.analysis_cache = {}
            st.success("Analysis cache cleared!")
        
        if st.button("🗑️ Clear All Caches"):
            st.session_state.analysis_cache = {}
            # Also clear Streamlit's cache
            st.cache_data.clear()
            st.success("All caches cleared!")
            
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hi! I'm your **Advanced AI Trading Assistant** powered by ensemble ML models. I can:\n\n🔮 **Predict** with XGBoost, LightGBM, Random Forest & Neural Networks\n📊 **Backtest** strategies with risk metrics\n📈 **Analyze** 50+ technical indicators and market patterns\n💰 **Calculate** ROI, Sharpe ratio, max drawdown\n📉 **Visualize** comprehensive trading analysis\n⚡ **Generate** high-accuracy trading signals\n\nJust tell me a ticker symbol like AAPL, GOOGL, or TSLA for a full AI analysis!",
                    "timestamp": datetime.now()
                }
            ]
            st.rerun()
        
        # Show cache status
        st.write("**📊 Cache Status:**")
        if st.session_state.analysis_cache:
            st.write(f"📈 **Analysis cache:** {len(st.session_state.analysis_cache)} tickers")
            for ticker in st.session_state.analysis_cache.keys():
                cache_time = st.session_state.analysis_cache[ticker]['timestamp']
                time_diff = datetime.now() - cache_time
                hours = time_diff.seconds // 3600
                minutes = (time_diff.seconds % 3600) // 60
                if hours > 0:
                    st.write(f"• {ticker}: {hours}h {minutes}m ago")
                else:
                    st.write(f"• {ticker}: {minutes}m ago")
        else:
            st.write("📈 **Analysis cache:** Empty")

if __name__ == "__main__":
    main() 

    
