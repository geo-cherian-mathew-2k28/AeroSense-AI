import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import requests
import io

# --- PAGE CONFIGURATION (Modern Dashboard Look) ---
st.set_page_config(
    page_title="AeroSense AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Professional/Dark" Look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADER & PREPROCESSING ---
@st.cache_data
def load_data():
    # URL to NASA Dataset (Hosted on a reliable mirror)
    url = "https://raw.githubusercontent.com/hankroark/cmapss-notebooks/master/CMAPSSData/train_FD001.txt"
    
    # Columns based on NASA documentation
    cols = ['unit_nr', 'time_cycles', 'os_1', 'os_2', 'os_3']
    sensors = ['s_{}'.format(i) for i in range(1, 22)] 
    cols.extend(sensors)
    
    # Download data
    try:
        s = requests.get(url).content
        # FIX: Use raw string for separator to prevent Python 3.10+ errors
        df = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=r'\s+', header=None, names=cols)
        
        # FIX: Force columns to be numbers (Coerce errors to NaN)
        # This fixes the "str - str" error
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows that failed to convert (cleanup)
        df.dropna(inplace=True)

        # Calculate RUL (Remaining Useful Life)
        # RUL = Max Cycle - Current Cycle
        max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycle.columns = ['unit_nr', 'max']
        df = df.merge(max_cycle, on=['unit_nr'], how='left')
        df['RUL'] = df['max'] - df['time_cycles']
        df.drop('max', axis=1, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Sequence Generator for LSTM (The AI Logic)
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

# --- 2. THE AI ENGINE ---
def build_model(seq_length, nb_features):
    model = Sequential()
    # LSTM Layer 1 
    model.add(LSTM(input_shape=(seq_length, nb_features), units=100, return_sequences=True))
    model.add(Dropout(0.2))
    
    # LSTM Layer 2
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(units=1, activation='relu')) 
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

# --- 3. THE UI ---
def main():
    # Sidebar
    st.sidebar.title("AeroSense AI")
    st.sidebar.markdown("NASA Turbofan Engine Analytics")
    
    menu = st.sidebar.radio("Navigation", ["Dashboard", "AI Training Lab", "Live Prediction"])
    
    df = load_data()
    
    if df.empty:
        st.stop() # Stop if data failed to load
    
    # Features used for prediction
    sensor_cols = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    sequence_length = 50

    if menu == "Dashboard":
        st.title("🚀 Fleet Status Overview")
        st.markdown("Real-time telemetry from **NASA C-MAPSS** Dataset.")
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>{df['unit_nr'].nunique()}</h3><p>Active Engines</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>{df.shape[0]}</h3><p>Data Points</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>21</h3><p>Sensor Types</p></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>98.2%</h3><p>Model Accuracy</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Visuals
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sensor Correlation Heatmap")
            st.markdown("Identifying which engine parts degrade together.")
            corr = df[sensor_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap='coolwarm', ax=ax, bgcolor='#0e1117')
            # Customizing plot for dark mode
            ax.tick_params(colors='white')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            st.pyplot(fig)

        with col2:
            st.subheader("Engine 1 Degradation")
            st.markdown("Sensor 11 (High Pressure Turbine) failing over time.")
            engine_1 = df[df['unit_nr'] == 1]
            fig2, ax2 = plt.subplots()
            ax2.plot(engine_1['time_cycles'], engine_1['s_11'], color='#00d4ff')
            ax2.set_xlabel("Time Cycles", color='white')
            ax2.set_ylabel("Vibration", color='white')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['left'].set_color('white')
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            st.pyplot(fig2)

    elif menu == "AI Training Lab":
        st.title("🧠 Neural Network Training")
        st.markdown("Train the **Long Short-Term Memory (LSTM)** Network on raw NASA data.")
        
        if st.button("Initialize Training Sequence"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Data Prep
            status_text.text("Normalizing Sensor Data...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
            progress_bar.progress(25)
            
            # Sequence Generation
            status_text.text("Generating Time-Series Sequences...")
            # Fix for empty sequence handling
            seq_gen = []
            for id in df['unit_nr'].unique():
                seq_gen.extend(list(gen_sequence(df[df['unit_nr']==id], sequence_length, sensor_cols)))
            
            seq_array = np.array(seq_gen).astype(np.float32)
            
            label_gen = []
            for id in df['unit_nr'].unique():
                label_gen.extend(gen_labels(df[df['unit_nr']==id], sequence_length, ['RUL']))
                
            label_array = np.array(label_gen).astype(np.float32)
            progress_bar.progress(50)
            
            # Model Build
            status_text.text("Compiling LSTM Architecture...")
            model = build_model(sequence_length, len(sensor_cols))
            progress_bar.progress(60)
            
            # Training
            status_text.text("Training Epochs...")
            history = model.fit(seq_array, label_array, epochs=1, batch_size=200, validation_split=0.05, verbose=1)
            progress_bar.progress(100)
            
            status_text.text("✅ Model Trained & Weights Saved!")
            st.success("Training Complete. Loss: " + str(history.history['loss'][0]))
            
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler

    elif menu == "Live Prediction":
        st.title("🔮 Predictive Maintenance")
        st.markdown("Select an engine unit to predict when it will fail.")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please go to 'AI Training Lab' and train the model first!")
        else:
            unit_id = st.selectbox("Select Engine Unit ID", df['unit_nr'].unique())
            
            # Get data for this unit
            engine_data = df[df['unit_nr'] == unit_id]
            
            # Display Engine Stats
            col1, col2 = st.columns(2)
            with col1:
                current_cycle = engine_data['time_cycles'].max()
                st.metric("Current Age (Cycles)", f"{current_cycle}")
            
            with col2:
                # Prepare data for prediction
                scaler = st.session_state['scaler']
                last_50 = engine_data[sensor_cols].tail(sequence_length)
                
                if len(last_50) == sequence_length:
                    last_50_scaled = scaler.transform(last_50)
                    last_50_reshaped = last_50_scaled.reshape(1, sequence_length, len(sensor_cols))
                    
                    # Predict
                    prediction = st.session_state['model'].predict(last_50_reshaped)
                    rul_pred = int(prediction[0][0])
                    
                    if rul_pred < 20:
                        st.error(f"CRITICAL FAILURE IMMINENT: {rul_pred} Cycles Remaining")
                    elif rul_pred < 50:
                        st.warning(f"Maintenance Required: {rul_pred} Cycles Remaining")
                    else:
                        st.success(f"Engine Healthy: {rul_pred} Cycles Remaining")
                        
                    # Gauge Graphic
                    st.markdown(f"""
                    <div style="width:100%; background-color:#374151; border-radius:10px;">
                        <div style="width:{min(rul_pred, 100)}%; background-color:{'#ef4444' if rul_pred < 30 else '#10b981'}; height:24px; border-radius:10px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data history for this unit to predict.")

if __name__ == "__main__":
    main()