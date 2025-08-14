import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        html, body, .main {
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease-in-out;
        }
        .aqi-card, .forecast-card, .metric-card, .city-card {
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        .aqi-card { padding: 2rem; text-align: center; }
        .forecast-card { padding: 1.5rem; text-align: center; }
        .metric-card, .city-card { padding: 1.5rem; text-align: center; }
        .centered-card { padding: 2rem; text-align: center; max-width: 600px; margin: 2rem auto; }
        /* Responsive */
        @media (max-width: 768px) {
            .centered-card { padding: 1rem; }
        }
        /* Light mode overrides */
        @media (prefers-color-scheme: light) {
            body, .main {
                background-color: #f8f9fa;
                color: #222;
            }
            .metric-card, .city-card, .aqi-card, .forecast-card {
                background: #fff;
                color: #222;
            }
        }
        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            body, .main {
                background-color: #181c1f;
                color: #eee;
            }
            .metric-card, .city-card, .aqi-card, .forecast-card {
                background: #23272b;
                color: #eee;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def add_custom_css():
    pass
