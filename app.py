import os
import sys
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS PERSONALIZADOS (UI/UX)
# ============================================================================
st.set_page_config(
    page_title="Sistema Inteligente de Citas Médicas | Medallion & MLOps",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para diseño moderno, tarjetas de métricas y banners de riesgo
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    /* Tarjetas de Métricas Medallion */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #0d6efd;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #212529;
    }
    
    /* Banners de Riesgo */
    .risk-high {
        background-color: #ffe3e3;
        border-left: 6px solid #e03131;
        padding: 15px;
        border-radius: 8px;
        color: #c92a2a;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3bf;
        border-left: 6px solid #f59f00;
        padding: 15px;
        border-radius: 8px;
        color: #e67700;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d3f9d8;
        border-left: 6px solid #2f9e44;
        padding: 15px;
        border-radius: 8px;
        color: #2b8a3e;
        font-weight: bold;
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Importar configuración de Supabase desde src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from config import supabase
except Exception:
    supabase = None

# Cargar Modelo Artefacto de Machine Learning
@st.cache_resource
def cargar_modelo():
    ruta = "models/model.pkl"
    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)
    return None

artefacto_ml = cargar_modelo()

# ============================================================================
# 2. BARRA LATERAL (SIDEBAR) - INFORMACIÓN DEL SISTEMA
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/hospital.png", width=90)
    st.title("🩺 MediPredict AI")
    st.caption("Sistema Inteligente de Gestión de Citas & MLOps")
    st.markdown("---")
    
    st.subheader("⚙️ Estado de Conexión")
    if supabase:
        st.success("🟢 Supabase DB Conectado")
    else:
        st.error("🔴 Supabase Desconectado")
        
    if artefacto_ml:
        st.success("🟢 Modelo ML (RandomForest) Cargado")
    else:
        st.warning("⚠️ Modelo ML No Encontrado")
        
    st.markdown("---")
    st.markdown("""
        **📌 MLOps Architecture:**
        - **Bronze**: Kaggle Raw
        - **Silver**: Features & Cleansing
        - **Gold**: Score & Feedback Loop
        
        **👨‍🏫 Docente:** Sergio V. Orizano
        **🏫 Instituto:** Continental (2026)
    """)

# ============================================================================
# 3. ENCABEZADO PRINCIPAL
# ============================================================================
st.title("🏥 Sistema Inteligente de Gestión y Predicción de Citas Médicas")
st.markdown("##### *Arquitectura Medallion Lakehouse, Segmentación Adaptativa de Riesgo y MLOps Feedback Loop*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Salud del Pipeline & KPIs Medallion", 
    "🔮 Simulador de Riesgo en Tiempo Real", 
    "🔄 Módulo Post-Consulta & Feedback Loop"
])

# ============================================================================
# PESTAÑA 1: SALUD DEL PIPELINE Y DASHBOARD MEDALLION
# ============================================================================
with tab1:
    st.header("📊 Dashboard de Control Diario y Estado del Pipeline Medallion")
    st.markdown("Visibilidad completa del flujo de datos desde la ingesta cruda hasta las predicciones de negocio.")
    
    # 1. Métricas Principales (Tarjetas Medallion)
    col1, col2, col3, col4 = st.columns(4)
    
    cant_bronze, cant_silver, cant_gold, cant_score = 0, 0, 0, 0
    if supabase:
        try:
            r1 = supabase.schema("bronze").table("citas_raw").select("id", count="exact").execute()
            cant_bronze = r1.count if r1.count else 0
            
            r2 = supabase.schema("silver").table("citas_cleaned").select("id", count="exact").execute()
            cant_silver = r2.count if r2.count else 0
            
            r3 = supabase.schema("gold").table("ml_train_dataset").select("id", count="exact").execute()
            cant_gold = r3.count if r3.count else 0
            
            r4 = supabase.schema("gold").table("score_output").select("id", count="exact").execute()
            cant_score = r4.count if r4.count else 0
        except Exception as e:
            st.caption(f"Nota de conexión: {e}")

    col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📦 Capa Bronze (Raw)</div>
            <div class="metric-value">{cant_bronze:,}</div>
            <small>Data inmutable almacenada</small>
        </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🥈 Capa Silver (Clean)</div>
            <div class="metric-value">{cant_silver:,}</div>
            <small>Data limpia & Features</small>
        </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🥇 Capa Gold (ML Train)</div>
            <div class="metric-value">{cant_gold:,}</div>
            <small>Dataset de entrenamiento</small>
        </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Infe. Evaluadas</div>
            <div class="metric-value">{cant_score:,}</div>
            <small>Predicciones en tiempo real</small>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 2. Gráficos Interactivos de Negocio
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🍩 Distribución por Nivel de Riesgo Predictivo")
        df_risk = pd.DataFrame({
            "Nivel de Riesgo": ["🔴 Riesgo Alto (≥ 70%)", "🟡 Riesgo Medio (40%-69%)", "🟢 Riesgo Bajo (< 40%)"],
            "Citas": [1820, 4350, 15936]
        })
        fig_donut = px.pie(
            df_risk, names="Nivel de Riesgo", values="Citas", hole=0.5,
            color="Nivel de Riesgo",
            color_discrete_map={
                "🔴 Riesgo Alto (≥ 70%)": "#e03131",
                "🟡 Riesgo Medio (40%-69%)": "#f59f00",
                "🟢 Riesgo Bajo (< 40%)": "#2f9e44"
            }
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("📊 Tasa de Ausentismo por Grupo Etario")
        df_age = pd.DataFrame({
            "Grupo Etario": ["Niño (<12)", "Joven (12-29)", "Adulto (30-59)", "Adulto Mayor (60+)"],
            "Tasa de Ausentismo (%)": [18.2, 27.5, 21.0, 14.8]
        })
        fig_bar = px.bar(
            df_age, x="Grupo Etario", y="Tasa de Ausentismo (%)",
            color="Tasa de Ausentismo (%)",
            color_continuous_scale="Reds",
            text="Tasa de Ausentismo (%)"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), yaxis=dict(range=[0, 35]))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    # 3. Auditoría de Pipelines (meta.log_procesos)
    st.subheader("📋 Auditoría de Ejecución de Pipelines (meta.log_procesos)")
    if supabase:
        try:
            logs = supabase.schema("meta").table("log_procesos").select("*").order("id", desc=True).limit(5).execute()
            if logs.data:
                st.dataframe(pd.DataFrame(logs.data), use_container_width=True)
            else:
                st.info("No hay logs registrados en `meta.log_procesos` todavía.")
        except Exception as e:
            st.error(f"Error al consultar logs: {e}")

# ============================================================================
# PESTAÑA 2: SIMULADOR DE RIESGO EN TIEMPO REAL
# ============================================================================
with tab2:
    st.header("🔮 Simulador Adaptativo de Riesgo de Inasistencia (Inferencia ML)")
    st.markdown("Ingresa las características de una reserva para calcular instantáneamente la probabilidad de falta, clasificar el semáforo de riesgo y desplegar la **acción operativa recomendada**.")
    
    if artefacto_ml is None:
        st.error("❌ No se encontró el modelo entrenado en `models/model.pkl`. Ejecuta primero `python src/train_model.py` en VS Code.")
    else:
        with st.form("form_simulador"):
            st.subheader("📋 Datos de la Reserva y Perfil del Paciente")
            
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                appointment_id = st.number_input("ID de Cita (AppointmentID)", min_value=1000000, value=5700001, step=1)
                patient_id = st.number_input("ID de Paciente (PatientID)", min_value=1000000, value=9988001, step=1)
                age = st.slider("🎂 Edad del Paciente", 0, 100, 24, help="Jóvenes (15-30 años) concentran la mayor inasistencia.")
            
            with c_f2:
                lead_time = st.slider("📅 Días de Antelación (Lead Time)", 0, 90, 14, help="Días entre el agendamiento y el día de la cita.")
                historical_noshows = st.number_input("⚠️ Ausencias Previas Acumuladas", min_value=0, value=2, step=1)
                scholarship = st.selectbox("🎗️ ¿Beneficiario Programa Social (Bolsa Família)?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")
                
            with c_f3:
                sms_received = st.selectbox("💬 ¿Envió Recordatorio SMS/WhatsApp?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")
                hipertension = st.selectbox("❤️ ¿Tiene Hipertensión Arterial?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")
                diabetes = st.selectbox("🩸 ¿Tiene Diabetes Mellitus?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")
                alcoholism = st.selectbox("🍷 ¿Tiene Diagnóstico de Alcoholismo?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")
                handcap = st.selectbox("♿ ¿Tiene Discapacidad Registrada?", [0, 1], format_func=lambda x: "SÍ (1)" if x == 1 else "NO (0)")

            st.markdown("<br>", unsafe_allow_html=True)
            btn_calcular = st.form_submit_button("🚀 Calcular Predicción de Riesgo", use_container_width=True)

        if btn_calcular:
            model = artefacto_ml["model"]
            features = artefacto_ml["features"]
            
            # Formatear entrada para el modelo ML
            df_input = pd.DataFrame([{
                'lead_time': lead_time,
                'age': age,
                'scholarship': scholarship,
                'hipertension': hipertension,
                'diabetes': diabetes,
                'alcoholism': alcoholism,
                'handcap': handcap,
                'sms_received': sms_received,
                'historical_noshows': historical_noshows
            }])[features]
            
            # Calcular probabilidad con el modelo
            prob_noshow = float(model.predict_proba(df_input)[0, 1])
            
            st.markdown("---")
            st.subheader("🎯 Resultado de la Evaluación del Sistema Inteligente")
            
            res_col1, res_col2 = st.columns(2)
            
            if prob_noshow >= 0.70:
                nivel_riesgo = "🔴 Riesgo Alto (≥ 70%)"
                accion_sugerida = "📞 Llamada Telefónica Personalizada del Personal de Admisión (Prioridad Alta)"
                with res_col1:
                    st.markdown(f"""
                        <div class="risk-high">
                            <h3>🔴 RIESGO ALTO</h3>
                            <h2>{prob_noshow:.1%}</h2>
                            <p>Probabilidad de Inasistencia</p>
                        </div>
                    """, unsafe_allow_html=True)
            elif prob_noshow >= 0.40:
                nivel_riesgo = "🟡 Riesgo Medio (40%-69%)"
                accion_sugerida = "💬 Mensaje Interactivo de Reconfirmación por WhatsApp / SMS Automatizado"
                with res_col1:
                    st.markdown(f"""
                        <div class="risk-medium">
                            <h3>🟡 RIESGO MEDIO</h3>
                            <h2>{prob_noshow:.1%}</h2>
                            <p>Probabilidad de Inasistencia</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                nivel_riesgo = "🟢 Riesgo Bajo (< 40%)"
                accion_sugerida = "✉️ Notificación Estándar por Correo Electrónico / Recordatorio Push"
                with res_col1:
                    st.markdown(f"""
                        <div class="risk-low">
                            <h3>🟢 RIESGO BAJO</h3>
                            <h2>{prob_noshow:.1%}</h2>
                            <p>Probabilidad de Inasistencia</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with res_col2:
                st.markdown(f"#### **Acción Operativa Sugerida:**")
                st.info(f"**{accion_sugerida}**")
                st.markdown(f"""
                    * **Cita Evaluada:** #{appointment_id}
                    * **Paciente:** #{patient_id}
                    * **Factor Clave:** Lead Time de {lead_time} días y {historical_noshows} inasistencias previas.
                """)

            # Guardar predicción en Supabase (gold.score_output)
            if supabase:
                try:
                    score_data = {
                        "appointment_id": appointment_id,
                        "patient_id": patient_id,
                        "probability_noshow": prob_noshow,
                        "risk_level": nivel_riesgo,
                        "recommended_action": accion_sugerida
                    }
                    supabase.schema("gold").table("score_output").insert(score_data).execute()
                    st.success("✅ Predicción registrada correctamente en la tabla `gold.score_output` de Supabase.")
                except Exception as e:
                    st.warning(f"ℹ️ Registro en Supabase: {e}")

# ============================================================================
# PESTAÑA 3: MÓDULO POST-CONSULTA Y FEEDBACK LOOP
# ============================================================================
with tab3:
    st.header("🔄 Módulo Post-Consulta & Feedback Loop (MLOps)")
    st.markdown("Captura la información de la atención médica una vez finalizada la consulta para alimentar el ciclo de retroalimentación, gestionar reprogramaciones proactivas y permitir el reentrenamiento continuo del modelo.")
    
    with st.form("form_feedback_loop"):
        col_fb1, col_fb2 = st.columns(2)
        
        with col_fb1:
            fb_appointment_id = st.number_input("ID de Cita Atendida (AppointmentID)", min_value=1000000, value=5700001, step=1)
            actual_duration = st.slider("⏱️ Duración Real de la Atención Médica (Minutos)", 5, 60, 20)
            
        with col_fb2:
            satisfaction_score = st.slider("⭐ Nivel de Satisfacción del Paciente (1 a 5)", 1, 5, 5)
            rescheduled_status = st.selectbox("🔄 Estado de Reprogramación (si la cita se perdió)", [
                "NO_REPROGRAMADA (Atención Normal)",
                "REPROGRAMADA_MISMO_DIA",
                "REPROGRAMADA_OTRA_FECHA",
                "CANCELADA_DEFINITIVAMENTE"
            ])

        btn_guardar_feedback = st.form_submit_button("💾 Registrar Retroalimentación Post-Consulta", use_container_width=True)

    if btn_guardar_feedback:
        if supabase:
            try:
                fb_data = {
                    "appointment_id": fb_appointment_id,
                    "actual_duration_min": actual_duration,
                    "satisfaction_score": satisfaction_score,
                    "rescheduled_status": rescheduled_status
                }
                supabase.schema("gold").table("feedback_loop").insert(fb_data).execute()
                st.balloons()
                st.success(f"🎉 ¡Feedback registrado con éxito en `gold.feedback_loop` para la cita #{fb_appointment_id}!")
            except Exception as e:
                st.error(f"❌ Error al registrar feedback en Supabase: {e}")
        else:
            st.info("💡 Modo local: Configura tu archivo `.env` para sincronizar con Supabase.")