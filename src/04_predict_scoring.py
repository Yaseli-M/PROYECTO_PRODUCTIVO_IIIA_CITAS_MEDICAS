import os
import sys
import pickle
import pandas as pd
import numpy as np

# Garantizar que Python reconozca los módulos internos de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

def registrar_log(proceso, capa, estado, registros=0, error=None):
    """Registra trazabilidad de auditoría en meta.log_procesos"""
    try:
        log_data = {
            "proceso_nombre": proceso,
            "capa_destino": capa,
            "registros_procesados": registros,
            "estado": estado,
            "mensaje_error": str(error) if error else None,
            "usuario_ejecutor": "PYTHON_SCRIPT"
        }
        supabase.schema("meta").table("log_procesos").insert(log_data).execute()
    except Exception as e:
        print(f" No se pudo registrar log en meta: {e}")

def cargar_modelo_campeon(ruta_modelo="models/champion_model.pkl"):
    """Carga el artefacto del modelo campeón guardado en MLOps"""
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f" No se encontró el modelo campeón en '{ruta_modelo}'. "
            "Ejecuta primero 'python src/03_compare_train_models.py'."
        )
    
    with open(ruta_modelo, "rb") as f:
        datos_modelo = pickle.load(f)
    
    nombre_modelo = datos_modelo.get('model_name', 'Desconocido')
    recall = datos_modelo.get('recall', 0.0)
    print(f" Modelo Campeón cargado con éxito: {nombre_modelo} (Recall: {recall:.4f})")
    
    return datos_modelo['model'], datos_modelo['features']

def obtener_citas_para_scoring():
    """Obtiene las citas de gold.score_input. Si está vacía, genera una muestra de simulación desde Silver."""
    print(" Consultando citas pendientes de evaluación en gold.score_input...")
    respuesta = supabase.schema("gold").table("score_input").select("*").execute()
    df_input = pd.DataFrame(respuesta.data)

    if df_input.empty:
        print(" 'gold.score_input' está vacía. Generando muestra de simulación desde silver.citas_cleaned...")
        resp_silver = supabase.schema("silver").table("citas_cleaned").select("*").limit(50).execute()
        df_silver = pd.DataFrame(resp_silver.data)
        
        if df_silver.empty:
            raise ValueError(" No hay datos en Silver para simular scoring.")
        
        columnas_input = [
            'appointment_id', 'patient_id', 'scheduled_day', 'appointment_day',
            'lead_time', 'age', 'scholarship', 'hipertension', 'diabetes',
            'alcoholism', 'handcap', 'sms_received', 'historical_noshows'
        ]
        df_input = df_silver[columnas_input].copy()
        
        # Insertar muestra en gold.score_input para poblar la tabla de entrada
        registros = df_input.where(pd.notnull(df_input), None).to_dict(orient='records')
        supabase.schema("gold").table("score_input").insert(registros).execute()
        print(f" Se insertaron {len(registros)} citas de prueba en gold.score_input.")

    return df_input

def ejecutar_scoring_predicciones():
    proceso_nombre = "PREDICCION_SCORING_GOLD"
    registrar_log(proceso_nombre, "GOLD", "INICIADO")

    try:
        # 1. Cargar el modelo campeón
        modelo, features = cargar_modelo_campeon()

        # 2. Obtener datos para la predicción
        df_input = obtener_citas_para_scoring()
        print(f"🔮 Generando predicciones de riesgo para {len(df_input)} citas médicas...")

        # 3. Extraer la matriz de características X
        X = df_input[features]

        # 4. Inferencia: Calcular probabilidad de inasistencia (clase 1)
        probabilidades = modelo.predict_proba(X)[:, 1]
        df_input['probability_noshow'] = np.round(probabilidades, 4)

        # 5. Regla de Negocio: Segmentar por Nivel de Riesgo y Acción Recomendada
        def categorizar_riesgo(prob):
            if prob >= 0.70:
                return '🔴 Riesgo Alto (≥ 70%)', '📞 Llamada Telefónica'
            elif prob >= 0.40:
                return '🟡 Riesgo Medio (40%-69%)', '💬 SMS/WhatsApp'
            else:
                return '🟢 Riesgo Bajo (< 40%)', '✉️ Correo Estándar'

        resultados = df_input['probability_noshow'].apply(categorizar_riesgo)
        df_input['risk_level'] = [r[0] for r in resultados]
        df_input['recommended_action'] = [r[1] for r in resultados]

        # 6. Preparar datos de salida para gold.score_output
        df_output = df_input[[
            'appointment_id', 'patient_id', 'probability_noshow', 'risk_level', 'recommended_action'
        ]].copy()

        # Sanear nulos para compatibilidad con JSON/Supabase
        df_output = df_output.where(pd.notnull(df_output), None)
        registros_output = df_output.to_dict(orient='records')

        # 7. Limpiar resultados previos e insertar nuevas predicciones en gold.score_output
        print(" Guardando predicciones en la tabla gold.score_output...")
        supabase.schema("gold").table("score_output").delete().neq("id", 0).execute()
        
        tamano_lote = 1000
        total = len(registros_output)
        for i in range(0, total, tamano_lote):
            lote = registros_output[i:i + tamano_lote]
            supabase.schema("gold").table("score_output").insert(lote).execute()

        print(" ¡Proceso de Inferencia y Scoring completado con éxito!")
        print("\n===  MUESTRA DE RESULTADOS DE PREDICCIÓN (gold.score_output) ===")
        print(df_output.head(10).to_string(index=False))

        registrar_log(proceso_nombre, "GOLD", "EXITOSO", registros=total)

    except Exception as e:
        print(f" Error durante el proceso de scoring: {e}")
        registrar_log(proceso_nombre, "GOLD", "ERROR", error=str(e))

if __name__ == "__main__":
    ejecutar_scoring_predicciones()