import os
import sys
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

def extraer_bronze_paginado(tamano_pagina=1000):
    """Extrae registros de bronze.citas_raw con paginación determinista"""
    print(" Leyendo datos desde bronze.citas_raw...")
    registros = []
    inicio = 0
    while True:
        respuesta = supabase.schema("bronze").table("citas_raw") \
            .select("*") \
            .order("id", desc=False) \
            .range(inicio, inicio + tamano_pagina - 1) \
            .execute()
        datos = respuesta.data
        if not datos:
            break
        registros.extend(datos)
        inicio += tamano_pagina
        if len(datos) < tamano_pagina:
            break
    print(f" Total leídos de Bronze: {len(registros)} registros.")
    return pd.DataFrame(registros)

def procesar_capa_silver():
    proceso_nombre = "ETL_BRONZE_A_SILVER"
    registrar_log(proceso_nombre, "SILVER", "INICIADO")
    
    try:
        df = extraer_bronze_paginado(tamano_pagina=1000)
        if df.empty:
            print(" No hay registros en Bronze para procesar.")
            return

        print("⚙️ Aplicando transformaciones e Ingeniería de Características...")
        df.columns = [col.lower() for col in df.columns]

        # 1. Conversión de fechas robusta
        df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce', utc=True)
        df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce', utc=True)

        # 2. Feature Engineering: Lead Time (Días de antelación)
        df['lead_time'] = (df['appointmentday'].dt.date - df['scheduledday'].dt.date).apply(
            lambda x: int(x.days) if pd.notnull(x) else 0
        )

        # 3. Mapeo Target (1 = Faltó / Yes, 0 = Asistió / No)
        df['no_show_target'] = df['no_show'].apply(
            lambda x: 1 if str(x).strip().upper() in ['YES', '1', 'TRUE'] else 0
        )

        # 4. Agrupar Edad
        condiciones_edad = [
            (df['age'] < 12),
            (df['age'] >= 12) & (df['age'] < 30),
            (df['age'] >= 30) & (df['age'] < 60),
            (df['age'] >= 60)
        ]
        etiquetas_edad = ['Niño', 'Joven', 'Adulto', 'Adulto Mayor']
        df['grupo_edad'] = np.select(condiciones_edad, etiquetas_edad, default='Desconocido')

        # 5. Auditoría de Calidad
        df['alerta_calidad_logica'] = (df['age'] < 0) | (df['lead_time'] < 0)

        # 6. Feature Engineering: Historial acumulado sin Data Leakage
        df = df.sort_values(by=['patientid', 'scheduledday'])
        df['historical_noshows'] = (
            df.groupby('patientid')['no_show_target'].cumsum() - df['no_show_target']
        ).astype(int)

        # 7. Mapeo final para silver.citas_cleaned
        df_silver = pd.DataFrame({
            'patient_id': df['patientid'].astype(int),
            'appointment_id': df['appointmentid'].astype(int),
            'gender': df['gender'],
            'scheduled_day': df['scheduledday'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'appointment_day': df['appointmentday'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'lead_time': df['lead_time'],
            'age': df['age'],
            'grupo_edad': df['grupo_edad'],
            'neighbourhood': df['neighbourhood'],
            'scholarship': df['scholarship'],
            'hipertension': df['hipertension'],
            'diabetes': df['diabetes'],
            'alcoholism': df['alcoholism'],
            'handcap': df['handcap'],
            'sms_received': df['sms_received'],
            'historical_noshows': df['historical_noshows'],
            'no_show_target': df['no_show_target'],
            'alerta_calidad_logica': df['alerta_calidad_logica'],
            'fuente_archivo': 'citas_medicas_raw.csv'
        })

        # Sanear nulos para compatibilidad con JSON/Supabase
        df_silver = df_silver.where(pd.notnull(df_silver), None)

        # 8. Carga a Supabase en lotes
        registros_silver = df_silver.to_dict(orient='records')
        tamano_lote = 1000
        total_registros = len(registros_silver)

        print(f" Insertando {total_registros} registros en silver.citas_cleaned...")
        for i in range(0, total_registros, tamano_lote):
            lote = registros_silver[i:i + tamano_lote]
            supabase.schema("silver").table("citas_cleaned").insert(lote).execute()
            print(f"   ➜ Lote {i // tamano_lote + 1} insertado ({min(i + tamano_lote, total_registros)}/{total_registros})")

        print(" ¡Transformación y carga a Capa Silver completada con éxito!")
        registrar_log(proceso_nombre, "SILVER", "EXITOSO", registros=total_registros)

    except Exception as e:
        print(f" Error durante el procesamiento de la Capa Silver: {e}")
        registrar_log(proceso_nombre, "SILVER", "ERROR", error=str(e))

if __name__ == "__main__":
    procesar_capa_silver()