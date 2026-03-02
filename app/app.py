st.markdown(
    """
<style>
  /* --- Page background --- */
  .stApp {
    background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 40%, #F3FAFF 100%);
  }

  /* --- Sidebar --- */
  section[data-testid="stSidebar"] {
    background: #EAF5FF !important;
    border-right: 1px solid rgba(11,31,53,0.12);
  }

  /* Make ALL text readable */
  html, body, [class*="css"]  {
    color: #0B1F35 !important;
  }

  /* Headings */
  h1, h2, h3, h4, h5, h6 {
    color: #0B3B66 !important;
  }

  /* Paragraph/help text */
  p, li, span, label, small {
    color: #0B1F35 !important;
  }

  /* Cards */
  .card {
    background: #FFFFFF;
    border: 1px solid rgba(11,31,53,0.12);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 20px rgba(11,31,53,0.06);
  }

  /* Buttons */
  .stButton>button {
    background: #2F80ED !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.0rem !important;
  }
  .stButton>button:hover {
    background: #1F6FDC !important;
    color: #FFFFFF !important;
  }

  /* --- Widget labels (the ones that became invisible) --- */
  div[data-testid="stWidgetLabel"] label,
  div[data-testid="stWidgetLabel"] p,
  .stRadio label, .stRadio p,
  .stSelectbox label, .stSelectbox p,
  .stNumberInput label, .stNumberInput p {
    color: #0B1F35 !important;
    font-weight: 600 !important;
  }

  /* --- Input boxes: force light background + dark text --- */
  div[data-baseweb="input"] input,
  div[data-baseweb="base-input"] input,
  div[data-baseweb="textarea"] textarea {
    background: #FFFFFF !important;
    color: #0B1F35 !important;
    border: 1px solid rgba(11,31,53,0.18) !important;
    border-radius: 10px !important;
  }

  /* Selectbox (closed state) */
  div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #0B1F35 !important;
    border: 1px solid rgba(11,31,53,0.18) !important;
    border-radius: 10px !important;
  }

  /* Selectbox dropdown menu - LIGHT TEXT ON DARK BACKGROUND */
  ul[role="listbox"] {
    background: #1F2937 !important;
  }
  ul[role="listbox"] li {
    color: #FFFFFF !important;
  }
  ul[role="listbox"] li:hover {
    background: #374151 !important;
    color: #FFFFFF !important;
  }

  /* Radio labels in sidebar */
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stRadio p {
    color: #0B1F35 !important;
    font-weight: 600 !important;
  }

  /* DataFrame container */
  div[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid rgba(11,31,53,0.12) !important;
  }

  /* Alerts */
  div[role="alert"] * {
    color: #0B1F35 !important;
  }
</style>
    """,
    unsafe_allow_html=True,
)
