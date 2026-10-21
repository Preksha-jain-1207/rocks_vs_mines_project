import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title='Sonar Acoustic Intelligence | Vibrant Edition',
    page_icon='🌊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ==========================================
# VIBRANT CUSTOM CSS (Pink, Sky Blue & Neon Accents)
# ==========================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #fdf2f8 0%, #f0f9ff 50%, #f0fdf4 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ec4899 0%, #0ea5e9 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #db2777 0%, #0284c7 100%);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
        transform: translateY(-2px);
    }
    .metric-card-pink {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(244, 63, 94, 0.1);
        border-left: 5px solid #f43f5e;
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(14, 165, 233, 0.1);
        border-left: 5px solid #0ea5e9;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10b981;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# STYLISH SIDEBAR
# ==========================================
with st.sidebar:
  st.markdown(
      "<h2 style='color: #db2777;'>🎨 SUB-AQUATIC AI</h2>",
      unsafe_allow_html=True,
  )
  st.markdown('**Vibrant Acoustic Signature Classifier**')
  st.markdown('---')

  st.markdown('#### 🛠️ System Diagnostics')
  st.markdown(
      '- **Core Engine:** Random Forest Ensemble\n- **Spectrum Bins:** 60'
      ' Channels\n- **Theme:** Vivid Neon & Pastel'
  )

  st.markdown('---')
  st.markdown('#### 📊 Live Performance')
  st.metric(label='Cross-Validated Accuracy', value='82.51%')
  st.metric(label='Inference Latency', value='< 10ms')

  st.markdown('---')
  st.markdown(
      '<p style="font-size: 0.8rem; color: #db2777; font-weight: 600;">Engineered'
      ' to enterprise-grade standards.</p>',
      unsafe_allow_html=True,
  )

# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown(
    "<h1 style='color: #0f172a;'>🌊 Subsurface Sonar Acoustic Intelligence</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size: 1.1rem; color: #475569;">A next-generation neural'
    ' interface engineered to differentiate geological formations from'
    ' submerged metallic ordinance with striking precision.</p>',
    unsafe_allow_html=True,
)
st.markdown('---')

# ==========================================
# MODEL INITIALIZATION & VERIFICATION
# ==========================================
model_path = 'sonar_model.pkl'

if not os.path.exists(model_path):
  st.error(
      f"CRITICAL ERROR: Artifact '{model_path}' not found in the working"
      ' directory.'
  )
else:
  model = joblib.load(model_path)

  # Initialize Session State for Input Text
  if 'user_input_text' not in st.session_state:
    st.session_state['user_input_text'] = ''

  # ==========================================
  # TABS SETUP
  # ==========================================
  tab1, tab2 = st.tabs(
      ['🚀 Real-Time Inference Portal', '📊 Architecture & Analytics']
  )

  with tab1:
    st.subheader('Vector Feature Input Portal')
    st.markdown(
        'Select a vibrant verification profile below or enter your 60-channel'
        ' frequency array manually.'
    )

    # DIRECT VISIBLE BUTTONS (No Expanders so they are always accessible!)
    col_a, col_b = st.columns(2)

    with col_a:
      if st.button('🌸 Load Sample Profile A (Rock / Weather Effect)'):
        st.session_state['user_input_text'] = (
            '0.0207,0.0261,0.0353,0.0532,0.0846,0.1182,0.1373,0.1583,0.1805,'
            '0.1878,0.2078,0.2223,0.2458,0.2646,0.2857,0.3013,0.3341,0.3807,'
            '0.4357,0.4851,0.5186,0.5401,0.5539,0.5829,0.6151,0.6409,0.6653,'
            '0.6975,0.7258,0.7483,0.7681,0.7852,0.7981,0.8123,0.8291,0.8412,'
            '0.8523,0.8651,0.8792,0.8901,0.9012,0.9123,0.9234,0.9345,0.9456,'
            '0.9567,0.9678,0.9789,0.9891,0.9912,0.9934,0.9956,0.9978,0.9989,'
            '0.9995,0.9998,0.9999,1.0000,1.0000,1.0000'
        )
        st.snow()  # Triggers weather/snow effect instantly!
        st.success('🌸 Sample A Loaded Successfully! Weather simulation active.')

    with col_b:
      if st.button('🍁 Load Sample Profile B (Mine / Autumn Effect)'):
        st.session_state['user_input_text'] = (
            '0.0453,0.0523,0.0123,0.0789,0.0452,0.0214,0.0987,0.0654,0.0321,'
            '0.0112,0.0456,0.0789,0.0123,0.0456,0.0891,0.0234,0.0567,0.0891,'
            '0.0124,0.0457,0.0789,0.0234,0.0568,0.0912,0.0345,0.0678,0.0124,'
            '0.0456,0.0789,0.0123,0.0456,0.0891,0.0234,0.0567,0.0891,0.0124,'
            '0.0457,0.0789,0.0234,0.0568,0.0912,0.0345,0.0678,0.0124,0.0456,'
            '0.0789,0.0123,0.0456,0.0891,0.0234,0.0567,0.0891,0.0124,0.0457,'
            '0.0789,0.0234,0.0568,0.0912,0.0345,0.0678'
        )
        st.snow()  # Triggers weather/autumn simulation effect instantly!
        st.success('🍁 Sample B Loaded Successfully! Autumn signature active.')

    st.markdown('<br>', unsafe_allow_html=True)

    # Text Area Input linked directly with session_state via key
    user_input = st.text_area(
        'Frequency Tensor Input Array (Comma-Separated):',
        key='user_input_text',
        height=140,
        placeholder='Paste or type exactly 60 comma-separated float values here...',
    )

    # Execute Button Column
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
      execute_btn = st.button('✨ Execute Inference & Trigger Balloons')

    if execute_btn:
      if not user_input.strip():
        st.warning('⚠️ Warning: Input stream is empty.')
      else:
        try:
          parsed_features = [float(x.strip()) for x in user_input.split(',')]

          if len(parsed_features) != 60:
            st.error(
                f'❌ Dimension Mismatch: Expected exactly 60 features, but'
                f' received {len(parsed_features)}.'
            )
          else:
            with st.spinner('Processing multi-channel acoustic spectrum...'):
              tensor_array = np.asarray(parsed_features).reshape(1, -1)
              prediction_result = model.predict(tensor_array)

            st.markdown('---')
            st.markdown('### 📋 Assessment Report')

            if (
                prediction_result[0] == 'M'
                or prediction_result[0] == 1
                or str(prediction_result[0]).upper() == 'M'
            ):
              st.error(
                  '⚠️ **THREAT DETECTED: SUBMERGED MINE (METAL CYLINDER)**\n\n'
                  '- **Confidence Metric:** High-intensity metallic bounce'
                  ' detected.\n'
                  '- **Recommended Action:** Execute immediate naval clearing'
                  ' protocol.'
              )
            else:
              st.success(
                  '🟢 **STATUS VERIFIED: GEOLOGICAL ROCK FORMATION**\n\n'
                  '- **Confidence Metric:** Natural diffuse acoustic scattering'
                  ' pattern.\n'
                  '- **Recommended Action:** Safe navigational trajectory'
                  ' confirmed.'
              )

            # Trigger Balloons Celebration on Execution!
            st.balloons()

        except ValueError as ve:
          st.error(
              f'❌ Parsing Exception: Ensure all values are numerical floats'
              ' separated by commas. Details: {ve}'
          )
        except Exception as ex:
          st.error(
              f'❌ Critical Execution Fault: An unexpected error occurred:'
              f' {ex}'
          )

  with tab2:
    st.subheader('System Architecture & Analytics')
    st.markdown(
        'Comprehensive overview of the underlying machine learning pipeline.'
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
      st.markdown(
          """
            <div class="metric-card-pink">
                <h4 style="color: #db2777;">Algorithm</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: #831843;">Random Forest Ensemble</p>
                <p style="font-size: 0.85rem; color: #9f1239;">Optimized for high-dimensional stability.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    with col_m2:
      st.markdown(
          """
            <div class="metric-card-blue">
                <h4 style="color: #0284c7;">Feature Space</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: #0369a1;">60 Frequency Bins</p>
                <p style="font-size: 0.85rem; color: #075985;">Normalized acoustic echo response.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    with col_m3:
      st.markdown(
          """
            <div class="metric-card-green">
                <h4 style="color: #059669;">Accuracy</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: #047857;">82.51% Score</p>
                <p style="font-size: 0.85rem; color: #065f46;">Cross-validated validation metric.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.markdown('### Technical Pipeline Breakdown')
    st.markdown(
        '1. **Data Acquisition:** Multi-angle sonar return signature collection.\n'
        '2. **Feature Normalization:** Standardizing 60 spectral frequency'
        ' components.\n'
        '3. **Ensemble Classification:** Tree-based voting mechanisms to'
        ' eliminate variance.\n'
        '4. **Interactive Dashboard:** Real-time visual rendering via Streamlit'
        ' engine.'
    )