import streamlit as st
import pandas as pd
import re
import string
from collections import Counter
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Procesamiento de Texto - NLP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS compacto para igualar las cajas, bordes y proporciones del wireframe al 100% de zoom
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 1rem !important;
    }
    .titulo-box {
        border: 2px dashed #ADB5BD;
        border-radius: 8px;
        padding: 12px 20px;
        text-align: center;
        margin-bottom: 12px;
        background-color: rgba(173, 181, 189, 0.05);
    }
    .titulo-box h2 {
        margin: 0;
        padding: 0;
        font-size: 1.55rem;
    }
    .titulo-box p {
        margin: 4px 0 0 0;
        font-size: 0.92rem;
        opacity: 0.85;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(173, 181, 189, 0.12);
        border: 1px solid rgba(173, 181, 189, 0.3);
        padding: 6px 12px;
        border-radius: 6px;
        text-align: center;
    }
    /* Altura compacta en vista normal, pero 100% expandible en Pantalla Completa */
    div[data-testid="stElementToolbar"] + div:not(:fullscreen) div[data-testid="stDataFrameResizable"] {
        max-height: 175px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def descargar_recursos_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

descargar_recursos_nltk()

# 1. BLOQUE SUPERIOR: TÍTULO Y DESCRIPCIÓN CENTRADOS (Igual al Wireframe)
st.markdown("""
<div class="titulo-box">
    <h2>Aplicación Web para Procesamiento de Texto (NLP)</h2>
    <p><b>Descripción breve e instrucciones generales de uso:</b> Suba un archivo CSV, ajuste los parámetros en el panel lateral y presione Ejecutar.</p>
</div>
""", unsafe_allow_html=True)

# 2. FILA DE INPUT CSV Y DISPLAY CSV EN CAJAS CON BORDE (Proporción 1:1 como en el Wireframe)
col_carga, col_display = st.columns([1, 1.05])

with col_carga:
    with st.container(border=True):
        st.markdown("**INPUT DE ARCHIVO CSV**")
        archivo_subido = st.file_uploader(
            "Arrastre o seleccione su archivo .csv",
            type=["csv"],
            label_visibility="collapsed"
        )

df_cargado = None
if archivo_subido is not None:
    df_cargado = pd.read_csv(archivo_subido)

with col_display:
    with st.container(border=True):
        st.markdown("**DISPLAY DEL ARCHIVO CSV SUBIDO**")
        if df_cargado is not None:
            st.dataframe(df_cargado, use_container_width=True)
        else:
            st.info("Vista previa de la tabla (Suba un archivo CSV a la izquierda).")

# 3. PANEL LATERAL (SIDEBAR) EN EL MISMO ORDEN EXACTO DEL WIREFRAME (1 al 5)
st.sidebar.markdown("### PANEL LATERAL (SIDEBAR)\n**Sección de Parámetros**")

if df_cargado is not None:
    columnas_disp = df_cargado.columns.tolist()
    idx_col = columnas_disp.index("texto") if "texto" in columnas_disp else 0
    columna_texto = st.sidebar.selectbox(
        "1. Columna de texto a procesar",
        options=columnas_disp,
        index=idx_col
    )
else:
    columna_texto = st.sidebar.selectbox(
        "1. Columna de texto a procesar",
        options=["texto"],
        index=0,
        disabled=True
    )

idioma_seleccionado = st.sidebar.selectbox(
    "2. Idioma de Stop-Words",
    options=["spanish", "english"],
    index=0
)

tokenizador_seleccionado = st.sidebar.selectbox(
    "3. Tokenizador",
    options=["TreebankWordTokenizer", "Expresión Regular (re.split)", "Espacios (.split)"],
    index=0
)

usar_stemming = st.sidebar.checkbox(
    "4. Aplicar Stemming (Snowball)",
    value=False
)

top_n_palabras = st.sidebar.slider(
    "5. Cantidad de palabras (Top N)",
    min_value=5,
    max_value=40,
    value=20,
    step=5
)

# 4. BOTÓN EJECUTAR DE ANCHO COMPLETO
boton_ejecutar = st.button(
    "EJECUTAR PROCESAMIENTO DE TEXTO",
    type="primary",
    disabled=(archivo_subido is None),
    use_container_width=True
)

def limpiar_y_tokenizar(texto, metodo_tokenizador, idioma, aplicar_stem):
    texto = str(texto).lower()
    texto = re.sub(r"[\n\t\r]+", " ", texto).strip()
    
    if metodo_tokenizador == "TreebankWordTokenizer":
        tokenizer = TreebankWordTokenizer()
        tokens_crudos = tokenizer.tokenize(texto)
    elif metodo_tokenizador == "Expresión Regular (re.split)":
        patron = re.split(r'([,.:;¿?_!¡"() \']|--|\s)', texto)
        tokens_crudos = [item.strip() for item in patron if item.strip()]
    else:
        tokens_crudos = texto.split()

    stopwords_set = set(nltk.corpus.stopwords.words(idioma))
    simbolos_extra = "¿¡“”«»‘’\"'—–-…•"
    puntuacion = set(string.punctuation).union(set(simbolos_extra)).union({"``", "''", "--"})
    caracteres_strip = string.punctuation + simbolos_extra + " \t\n\r"
    
    stemmer = SnowballStemmer(idioma) if aplicar_stem else None
    
    tokens_procesados = []
    for tok in tokens_crudos:
        tok_limpio = tok.strip(caracteres_strip)
        if tok_limpio and tok_limpio.isalpha() and tok_limpio not in puntuacion and tok_limpio not in stopwords_set:
            if stemmer is not None:
                tok_limpio = stemmer.stem(tok_limpio)
            tokens_procesados.append(tok_limpio)
            
    return tokens_procesados

# 5. SECCIONES DE RESULTADOS EN DOS COLUMNAS CON BORDES (Igual al Wireframe)
if boton_ejecutar and df_cargado is not None:
    lista_documentos = df_cargado[columna_texto].dropna().astype(str).tolist()
    
    conteo_tokens_por_doc = []
    for idx, doc in enumerate(lista_documentos):
        tokens_doc = limpiar_y_tokenizar(doc, tokenizador_seleccionado, idioma_seleccionado, usar_stemming)
        titulo_doc = df_cargado["titulo"].iloc[idx] if "titulo" in df_cargado.columns else f"Doc {idx}"
        conteo_tokens_por_doc.append({
            "Doc ID": idx,
            "Título": titulo_doc,
            "N° Tokens": len(tokens_doc)
        })
        
    texto_unido = " ".join(lista_documentos)
    tokens_totales = limpiar_y_tokenizar(texto_unido, tokenizador_seleccionado, idioma_seleccionado, usar_stemming)
    
    bolsa_de_palabras = Counter(tokens_totales)
    lexico = sorted(bolsa_de_palabras.keys())
    
    col_estadisticas, col_plotting = st.columns([1, 1.05])
    
    # COLUMNA IZQUIERDA: SECCIÓN 1 ESTADÍSTICAS
    with col_estadisticas:
        with st.container(border=True):
            st.markdown("##### SECCIÓN 1: ESTADÍSTICAS")
            
            # 2 tarjetas de métricas arriba (como en el wireframe)
            m1, m2 = st.columns(2)
            m1.metric("Tamaño Léxico", len(lexico))
            m2.metric("Total Tokens", len(tokens_totales))
            
            # Tabla 1: Total de tokens por documento (al medio, igual que en el wireframe)
            st.markdown("**Total de tokens por documento**")
            df_tokens_doc = pd.DataFrame(conteo_tokens_por_doc)
            st.dataframe(df_tokens_doc, hide_index=True, use_container_width=True)
            
            # Tabla 2: Primeras 15 palabras del léxico (abajo, igual que en el wireframe)
            st.markdown("**Primeras 15 palabras del léxico**")
            df_lexico_15 = pd.DataFrame({
                "Índice": range(1, min(16, len(lexico) + 1)),
                "Palabra (Token)": lexico[:15],
                "Frecuencia": [bolsa_de_palabras[w] for w in lexico[:15]]
            })
            st.dataframe(df_lexico_15, hide_index=True, use_container_width=True)

    # COLUMNA DERECHA: SECCIÓN 2 PLOTTING (Ambos gráficos visibles uno sobre otro)
    with col_plotting:
        with st.container(border=True):
            st.markdown("##### SECCIÓN 2: PLOTTING (GRÁFICOS)")
            
            principales = bolsa_de_palabras.most_common(top_n_palabras)
            palabras = [palabra for palabra, frec in principales]
            conteos = [frec for palabra, frec in principales]
            
            # Gráfico 1 arriba: Barras Horizontales (Matplotlib)
            st.markdown("**Gráfico 1: Barras Horizontales (Matplotlib)**")
            fig_bar, ax = plt.subplots(figsize=(6.5, 2.3), dpi=120)
            barras = ax.barh(palabras[:10], conteos[:10], color="#CC5DE8")
            ax.bar_label(barras, padding=3, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Apariciones", fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            ax.margins(x=0.15)
            plt.tight_layout()
            st.pyplot(fig_bar, clear_figure=True, use_container_width=True)
            
            # Gráfico 2 abajo: Treemap Interactivo (Plotly)
            st.markdown("**Gráfico 2: Treemap Interactivo (Plotly)**")
            fig_tree = px.treemap(
                names=palabras,
                parents=[""] * len(palabras),
                values=conteos,
                color=conteos,
                color_continuous_scale="bugn_r"
            )
            fig_tree.update_traces(textinfo="label+value")
            fig_tree.update_layout(
                margin=dict(t=5, l=5, r=5, b=5),
                height=215
            )
            st.plotly_chart(fig_tree, use_container_width=True)