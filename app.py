
# Despliegue

# 1.   Cargar el modelo
# 2.   Cargar datos futuros
# 3.   Preparar los datos futuros
# 3.   Aplicar el modelo para la predicción


# Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el modelo
import pickle
filename = 'modelo-NN.pkl'
modelo, min_max_scaler, variables = pickle.load(open(filename, 'rb'))

# Cargamos los datos futuros
# data = pd.read_csv("videojuegos-datosFuturos.csv")
# data.head()

# Alternativa
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Predicción de Inversión – Videojuegos",
    page_icon="🎮",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

/* ── Root tokens ─────────────────────────────────── */
:root {
    --primary: #01696f;
    --primary-light: #cedcd8;
    --bg: #f7f6f2;
    --surface: #ffffff;
    --text: #28251d;
    --muted: #7a7974;
    --border: rgba(40,37,29,.10);
    --radius: 12px;
    --shadow: 0 4px 24px rgba(1,105,111,.08);
}

/* ── Global ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Satoshi', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 640px !important;
    padding: 2.5rem 2rem 3rem !important;
}

/* ── Hero header ─────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
    border-radius: var(--radius);
    padding: 2rem 2rem 1.6rem;
    margin-bottom: 1.75rem;
    color: #fff;
    box-shadow: var(--shadow);
}
.hero-icon { font-size: 2.4rem; margin-bottom: .4rem; }
.hero h1 {
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0 0 .35rem;
    line-height: 1.2;
    color: #fff;
}
.hero p {
    font-size: .875rem;
    opacity: .75;
    margin: 0;
    color: #fff;
}

/* ── Section card ────────────────────────────────── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 12px rgba(40,37,29,.05);
}
.section-title {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: .45rem;
}
.section-title span { font-size: .95rem; }

/* ── Streamlit widget overrides ──────────────────── */
div[data-testid="stSlider"] > div > div > div {
    background: var(--primary-light) !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--primary) !important;
}
.stSelectbox > div > div,
.stSlider {
    margin-bottom: .1rem !important;
}

/* ── Labels ──────────────────────────────────────── */
label, .stLabel {
    font-size: .825rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin-bottom: .2rem !important;
}

/* ── Predict button ──────────────────────────────── */
.stButton > button {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .65rem 2rem !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background .18s ease !important;
    cursor: pointer !important;
    margin-top: .5rem;
}
.stButton > button:hover {
    background: #0c4e54 !important;
}

/* ── Result box ──────────────────────────────────── */
.result-box {
    background: #eaf4f4;
    border: 1.5px solid var(--primary-light);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-icon { font-size: 2rem; }
.result-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: .15rem;
}
.result-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
}

/* ── Data preview table ──────────────────────────── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Divider ─────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🎮</div>
    <h1>Predicción de Inversión</h1>
    <p>Tienda de videojuegos · Modelo de clasificación</p>
</div>
""", unsafe_allow_html=True)

# ── Sección: Perfil del jugador ───────────────────────────────────────────────
st.markdown("""
<div class="section-card">
    <div class="section-title"><span>👤</span> Perfil del jugador</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    Edad = st.slider("Edad", min_value=14, max_value=52, value=20, step=1)
    Sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])

with col2:
    Consumidor_habitual = st.selectbox(
        "Consumidor habitual",
        ["Sí", "No"],
    )
    Plataforma = st.selectbox(
        "Plataforma",
        ["Play Station", "Xbox", "PC", "Otros"],
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Sección: Preferencia de juego ────────────────────────────────────────────
st.markdown("""
<div class="section-card">
    <div class="section-title"><span>🕹️</span> Preferencia de juego</div>
""", unsafe_allow_html=True)

videojuego = st.selectbox(
    "Videojuego",
    [
        "Mass Effect", "Battlefield", "Fifa", "KOA: Reckoning",
        "Crysis", "Sim City", "Dead Space", "F1",
    ],
)

st.markdown("</div>", unsafe_allow_html=True)

# ── Botón de predicción ───────────────────────────────────────────────────────
predict = st.button("🔮  Predecir inversión")

# ── Resultado ─────────────────────────────────────────────────────────────────
if predict:
    consumidor_bool = "True" if Consumidor_habitual == "Sí" else "False"
    datos = [[Edad, f"'{videojuego}'", f"'{Plataforma}'", Sexo, consumidor_bool]]
    data = pd.DataFrame(
        datos,
        columns=["Edad", "videojuego", "Plataforma", "Sexo", "Consumidor_habitual"],
    )
    # Se realiza la preparación
    data_preparada=data.copy()

    # En despliegue drop_first= False
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma','Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
    data_preparada.head()
    
    '''
    Se adicionan las columnas faltantes, ya que cuando se hacen las dummies a los datos futuros
    quedan dummies faltantes; por ejemplo faltan juegos como fifa, f1, etc.
    '''
    
    # Se adicionan las columnas faltantes
    data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
    data_preparada.head()
    
    # Se normaliza la edad para predecir con Knn, Red, SVM
    # En los despliegues no se llama fit
    data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
    data_preparada.head()
    
    # Hacemos la predicción con el Tree
    Y_pred = modelo.predict(data_preparada)
    print(Y_pred)
    
    data['Prediccion']=Y_pred
    data.head()
    
    data

    st.markdown("""
    <div class="result-box">
        <div class="result-icon">📊</div>
        <div>
            <div class="result-label">Datos capturados</div>
            <div class="result-value">Listos para el modelo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Vista previa de los datos**")
    st.dataframe(data, use_container_width=True, hide_index=True)

    # Recordar medida de error del modelo

    st.warning("El modelo tiene un error del 10%")


