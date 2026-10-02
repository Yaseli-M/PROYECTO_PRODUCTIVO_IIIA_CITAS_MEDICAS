import os
import sys
import pandas as pd
import numpy as np

# Garantizar que Python encuentre los módulos dentro de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import supabase

def registrar_log(proceso, capa, estado, registros=0, error=None):
    """Registra eventos en la tabla meta.log_procesos"""
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
    """Extrae todos los registros de bronze.citas_raw paginando de 1000 en 1000"""
    print(" Leyendo datos desde bronze.citas_raw...")
    registros = []
    inicio = 0
    while True:
        respuesta = supabase.schema("bronze").table("citas_raw") \
            .select("*") \
            .range(inicio, inicio + tamano_pagina - 1) \
            .execute()
        
        datos = respuesta.data
        if not datos:
            break
        registros.extend(datos)
        print(f"  ➜ Registros acumulados: {len(registros)}...")
        inicio += tamano_pagina
        if len(datos) < tamano_pagina:
            break
    
    print(f" Lectura completa: Se leyeron {len(registros)} registros de la Capa Bronze.")
    return pd.DataFrame(registros)

def procesar_capa_silver():
    proceso_nombre = "ETL_BRONZE_A_SILVER"
    registrar_log(proceso_nombre, "SILVER", "INICIADO")
    
    try:
        # 1. Extraer data de Bronze en bloques de 1000
        df = extraer_bronze_paginado(tamano_pagina=1000)
        if df.empty:
            print(" No hay registros en Bronze para procesar.")
            return

        print(" Aplicando transformaciones e Ingeniería de Características...")

        # Asegurar nombres de columnas en minúsculas
        df.columns = [col.lower() for col in df.columns]

        # 2. Convertir fechas de forma robusta con utc=True
        df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce', utc=True)
        df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce', utc=True)

        # 3. Calcular Lead Time (Días de antelación)
        df['lead_time'] = (df['appointmentday'].dt.date - df['scheduledday'].dt.date).apply(lambda x: x.days if pd.notnull(x) else 0)

        # 4. Mapear Target No_show (1 = Faltó / Yes, 0 = Asistió / No)
        df['no_show_target'] = df['no_show'].apply(lambda x: 1 if str(x).strip().upper() in ['YES', '1', 'TRUE'] else 0)

        # 5. Agrupar Edad
        condiciones_edad = [
            (df['age'] < 12),
            (df['age'] >= 12) & (df['age'] < 30),
            (df['age'] >= 30) & (df['age'] < 60),
            (df['age'] >= 60)
        ]
        etiquetas_edad = ['Niño', 'Joven', 'Adulto', 'Adulto Mayor']
        df['grupo_edad'] = np.select(condiciones_edad, etiquetas_edad, default='Desconocido')

        # 6. Flag de Alerta de Calidad Lógica (edades negativas o lead_time negativo)
        df['alerta_calidad_logica'] = (df['age'] < 0) | (df['lead_time'] < 0)

        # 7. Calcular Historial Acumulado de Inasistencias por Paciente (historical_noshows)
        df = df.sort_values(by=['patientid', 'scheduledday'])
        df['historical_noshows'] = df.groupby('patientid')['no_show_target'].cumsum() - df['no_show_target']

        # 8. Mapear nombres de columnas finales para silver.citas_cleaned
        df_silver = pd.DataFrame({
            'patient_id': df['patientid'],
            'appointment_id': df['appointmentid'],
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
            'fuente_archivo': 'KaggleV2-May-2016.csv'
        })

        # 9. Cargar en Lotes a silver.citas_cleaned
        registros_silver = df_silver.to_dict(orient='records')
        tamano_lote = 1000
        total_registros = len(registros_silver)

        print(f" Insertando {total_registros} registros en la tabla silver.citas_cleaned...")
        for i in range(0, total_registros, tamano_lote):
            lote = registros_silver[i:i+tamano_lote]
            supabase.schema("silver").table("citas_cleaned").insert(lote).execute()
            print(f" Lote {i // tamano_lote + 1} insertado ({min(i + tamano_lote, total_registros)}/{total_registros})")

        print(" ¡Procesamiento a la Capa Silver completado con éxito!")
        registrar_log(proceso_nombre, "SILVER", "EXITOSO", registros=total_registros)

    except Exception as e:
        print(f" Error durante el procesamiento de la Capa Silver: {e}")
        registrar_log(proceso_nombre, "SILVER", "ERROR", error=str(e))

if __name__ == "__main__":
    procesar_capa_silver()
