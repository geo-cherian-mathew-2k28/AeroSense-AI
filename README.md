<!-- 
  AEROSENSE PRO UI README 
  Single File. Copy-Paste this entire block into your README.md
-->

<div align="center">
  
  <!-- HERO BANNER -->
  <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 50px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 20px 50px rgba(0,0,0,0.5);">
    <h1 style="color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 50px; font-weight: 800; letter-spacing: 2px; margin: 0;">
      🚀 AEROSENSE <span style="color: #00d4ff;">AI</span>
    </h1>
    <p style="color: #8b949e; font-family: 'Consolas', monospace; font-size: 16px; letter-spacing: 1px; margin-top: 10px;">
      PREDICTIVE MAINTENANCE MISSION CONTROL
    </p>
    <br>
    <div style="display: flex; justify-content: center; gap: 10px;">
      <img src="https://img.shields.io/badge/BUILD-PASSING-success?style=for-the-badge&logo=github-actions&color=2ea44f" height="28">
      <img src="https://img.shields.io/badge/AI-TENSORFLOW-orange?style=for-the-badge&logo=tensorflow&logoColor=white" height="28">
      <img src="https://img.shields.io/badge/INTERFACE-STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" height="28">
      <img src="https://img.shields.io/badge/LICENSE-MIT-blue?style=for-the-badge" height="28">
    </div>
  </div>

</div>

<br>

<!-- DASHBOARD CARDS ROW -->
<table width="100%" style="border-collapse: separate; border-spacing: 15px; border: none;">
  <tr>
    <!-- CARD 1 -->
    <td width="33%" style="background-color: #0d1117; border-radius: 10px; border: 1px solid #30363d; padding: 25px; vertical-align: top; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
      <h3 style="color: #e6edf3; margin-top: 0; font-family: sans-serif;">🧠 THE BRAIN</h3>
      <p style="color: #8b949e; font-size: 14px; line-height: 1.5;">Deep Learning engine trained on NASA Telemetry data to detect subtle degradation.</p>
      <hr style="border-color: #30363d; margin: 15px 0;">
      <div style="font-family: monospace; font-size: 13px;">
        <span style="color: #ff7b72;">Model:</span> <span style="color: #c9d1d9;">LSTM (Recurrent)</span><br>
        <span style="color: #ff7b72;">Input:</span> <span style="color: #c9d1d9;">Time-Series (50)</span><br>
        <span style="color: #ff7b72;">Metric:</span> <span style="color: #c9d1d9;">RMSE / RUL Accuracy</span>
      </div>
    </td>
    <!-- CARD 2 -->
    <td width="33%" style="background-color: #0d1117; border-radius: 10px; border: 1px solid #30363d; padding: 25px; vertical-align: top; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
      <h3 style="color: #e6edf3; margin-top: 0; font-family: sans-serif;">📊 THE VIEW</h3>
      <p style="color: #8b949e; font-size: 14px; line-height: 1.5;">Interactive "Mission Control" dashboard for real-time fleet monitoring.</p>
      <hr style="border-color: #30363d; margin: 15px 0;">
      <div style="font-family: monospace; font-size: 13px;">
        <span style="color: #79c0ff;">Viz 1:</span> <span style="color: #c9d1d9;">Parallel Coordinates</span><br>
        <span style="color: #79c0ff;">Viz 2:</span> <span style="color: #c9d1d9;">Real-time Gauges</span><br>
        <span style="color: #79c0ff;">Viz 3:</span> <span style="color: #c9d1d9;">3D Anomaly Plot</span>
      </div>
    </td>
    <!-- CARD 3 -->
    <td width="33%" style="background-color: #0d1117; border-radius: 10px; border: 1px solid #30363d; padding: 25px; vertical-align: top; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
      <h3 style="color: #e6edf3; margin-top: 0; font-family: sans-serif;">⚡ THE CORE</h3>
      <p style="color: #8b949e; font-size: 14px; line-height: 1.5;">Engineered for reliability with offline-first architecture and synthetic fallbacks.</p>
      <hr style="border-color: #30363d; margin: 15px 0;">
      <div style="font-family: monospace; font-size: 13px;">
        <span style="color: #d2a8ff;">Cache:</span> <span style="color: #c9d1d9;">Local Priority</span><br>
        <span style="color: #d2a8ff;">Backup:</span> <span style="color: #c9d1d9;">Synthetic Generator</span><br>
        <span style="color: #d2a8ff;">Deploy:</span> <span style="color: #c9d1d9;">1-Click Launch</span>
      </div>
    </td>
  </tr>
</table>

<br>

<!-- DATA FLOW DIAGRAM (VISUAL HTML VERSION) -->
<div align="center">
  <h2 style="color: #c9d1d9; font-family: sans-serif; border-bottom: 1px solid #30363d; display: inline-block; padding-bottom: 10px;">⚙️ SYSTEM ARCHITECTURE</h2>
</div>

<div align="center" style="background-color: #0d1117; padding: 30px; border-radius: 12px; border: 1px solid #30363d; margin-top: 10px;">
  
  <!-- Row 1: Source -->
  <div style="display: inline-block; background: #238636; color: white; padding: 10px 20px; border-radius: 20px; font-family: sans-serif; font-weight: bold; box-shadow: 0 4px 10px rgba(35, 134, 54, 0.4);">
    🛰️ NASA C-MAPSS DATA
  </div>
  
  <div style="font-size: 24px; color: #8b949e; margin: 5px 0;">⬇</div>
  
  <!-- Row 2: Processing -->
  <div style="display: flex; justify-content: center; gap: 20px;">
    <div style="background: #1f6feb; color: white; padding: 8px 15px; border-radius: 8px; font-family: monospace; font-size: 14px;">📂 Local Cache</div>
    <div style="background: #1f6feb; color: white; padding: 8px 15px; border-radius: 8px; font-family: monospace; font-size: 14px;">☁️ Web Mirror</div>
    <div style="background: #9e6a03; color: white; padding: 8px 15px; border-radius: 8px; font-family: monospace; font-size: 14px;">⚠️ Synthetic Gen</div>
  </div>

  <div style="font-size: 24px; color: #8b949e; margin: 5px 0;">⬇</div>

  <!-- Row 3: The Brain -->
  <div style="display: inline-block; background: #d2a8ff; color: #000; padding: 12px 25px; border-radius: 8px; font-family: sans-serif; font-weight: bold; border: 2px solid #a371f7;">
    🧠 LSTM DEEP LEARNING MODEL
  </div>

  <div style="font-size: 24px; color: #8b949e; margin: 5px 0;">⬇</div>

  <!-- Row 4: The Output -->
  <div style="display: inline-block; background: linear-gradient(90deg, #00d4ff, #005bea); color: white; padding: 15px 30px; border-radius: 30px; font-family: sans-serif; font-weight: 900; letter-spacing: 1px; box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);">
    🚀 MISSION CONTROL DASHBOARD
  </div>

</div>

<br>

<!-- TERMINAL INSTALLATION GUIDE -->
<div align="center">
  <h2 style="color: #c9d1d9; font-family: sans-serif; border-bottom: 1px solid #30363d; display: inline-block; padding-bottom: 10px;">🛠️ DEPLOYMENT PROTOCOL</h2>
</div>

<div style="background-color: #0d1117; border-radius: 8px; border: 1px solid #30363d; box-shadow: 0 10px 30px rgba(0,0,0,0.5); overflow: hidden; font-family: 'Consolas', 'Courier New', monospace; max-width: 800px; margin: 0 auto;">
  
  <!-- Terminal Header -->
  <div style="background-color: #161b22; padding: 10px 15px; display: flex; align-items: center; border-bottom: 1px solid #30363d;">
    <div style="width: 12px; height: 12px; background-color: #ff5f56; border-radius: 50%; margin-right: 8px;"></div>
    <div style="width: 12px; height: 12px; background-color: #ffbd2e; border-radius: 50%; margin-right: 8px;"></div>
    <div style="width: 12px; height: 12px; background-color: #27c93f; border-radius: 50%;"></div>
    <span style="margin-left: 10px; color: #8b949e; font-size: 12px;">admin@nasa-lab:~/aerosense</span>
  </div>

  <!-- Terminal Body -->
  <div style="padding: 25px; text-align: left;">
    <p style="margin: 0; color: #8b949e;"># 1. Clone the repository</p>
    <p style="margin: 5px 0 20px 0; color: #e6edf3;">
      <span style="color: #79c0ff;">$</span> git clone https://github.com/YOUR_USERNAME/AeroSense-AI.git
    </p>

   <p style="margin: 0; color: #8b949e;"># 2. Install Python dependencies</p>
    <p style="margin: 5px 0 20px 0; color: #e6edf3;">
      <span style="color: #79c0ff;">$</span> pip install -r requirements.txt
    </p>

   <p style="margin: 0; color: #8b949e;"># 3. Launch Mission Control</p>
    <p style="margin: 5px 0 20px 0; color: #e6edf3;">
      <span style="color: #79c0ff;">$</span> streamlit run dashboard.py
    </p>

  <p style="margin: 25px 0 0 0; color: #3fb950; font-weight: bold;">
      > [SYSTEM ONLINE] Dashboard active on http://localhost:8501
    </p>
  </div>
</div>

<br>

<!-- FILE STRUCTURE EXPLORER -->
<div align="center">
<table width="100%" style="border: 1px solid #30363d; border-radius: 8px; border-spacing: 0; overflow: hidden; max-width: 800px;">
  <tr style="background-color: #161b22;">
    <th align="left" style="padding: 12px; color: #e6edf3; border-bottom: 1px solid #30363d; font-family: sans-serif;">📂 REPO MANIFEST</th>
    <th align="left" style="padding: 12px; color: #e6edf3; border-bottom: 1px solid #30363d; font-family: sans-serif;">DESCRIPTION</th>
  </tr>
  <tr>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #79c0ff; font-family: monospace;">📄 dashboard.py</td>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #8b949e; font-size: 14px;">Main Application GUI (Streamlit)</td>
  </tr>
  <tr>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #ff7b72; font-family: monospace;">📄 train.py</td>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #8b949e; font-size: 14px;">AI Training Pipeline Script</td>
  </tr>
  <tr>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #d2a8ff; font-family: monospace;">🧠 my_model.keras</td>
    <td style="padding: 12px; border-bottom: 1px solid #21262d; color: #8b949e; font-size: 14px;">Trained Neural Network File</td>
  </tr>
  <tr>
    <td style="padding: 12px; color: #e3b341; font-family: monospace;">📦 requirements.txt</td>
    <td style="padding: 12px; color: #8b949e; font-size: 14px;">Dependency Manifest</td>
  </tr>
</table>
</div>

<br>

<div align="center">
  <p style="color: #666; font-size: 12px; font-family: sans-serif;">
    Engineered by <b>Geo Cherian Mathew</b> | Powered by NASA Open Data | 
  </p>
</div>
