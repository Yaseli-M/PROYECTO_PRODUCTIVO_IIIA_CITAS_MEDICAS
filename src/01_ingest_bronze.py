import os
import sys
import pandas as pd
import numpy as np

# Garantizar que Python reconozca los módulos internos de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import supabase

def registrar_log(proceso, capa, estado, registros=0, error=None):
    """Registra eventos de auditoría en la tabla meta.log_procesos"""
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

def cargar_csv_a_bronze(ruta_csv="data/raw/citas_medicas_raw.csv"):
    """
    Carga inmutable de los datos crudos a la tabla bronze.citas_raw en Supabase.
    """
    proceso_nombre = "INGESTA_CSV_A_BRONZE"
    registrar_log(proceso_nombre, "BRONZE", "INICIADO")

    try:
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f" No se encontró el archivo en la ruta: {ruta_csv}")

        print(f" Leyendo dataset crudo desde: {ruta_csv}")
        df = pd.read_csv(ruta_csv)

        # 1. Renombrar columnas a minúsculas estándar
        df = df.rename(columns={
            'PatientId': 'patientid',
            'AppointmentID': 'appointmentid',
            'Gender': 'gender',
            'ScheduledDay': 'scheduledday',
            'AppointmentDay': 'appointmentday',
            'Age': 'age',
            'Neighbourhood': 'neighbourhood',
            'Scholarship': 'scholarship',
            'Hipertension': 'hipertension',
            'Diabetes': 'diabetes',
            'Alcoholism': 'alcoholism',
            'Handcap': 'handcap',
            'SMS_received': 'sms_received',
            'No-show': 'no_show'
        })

        # 2. Agregar metadata de origen y sanear nulos
        df['fuente_archivo'] = os.path.basename(ruta_csv)
        df = df.where(pd.notnull(df), None)

        # 3. Cargar en lotes de 1000 a Supabase
        registros = df.to_dict(orient='records')
        tamano_lote = 1000
        total = len(registros)

        print(f" Cargando {total} registros en la Capa Bronze (bronze.citas_raw)...")
        for i in range(0, total, tamano_lote):
            lote = registros[i:i + tamano_lote]
            supabase.schema("bronze").table("citas_raw").insert(lote).execute()
            print(f"   ➜ Lote {i // tamano_lote + 1} insertado ({min(i + tamano_lote, total)}/{total})")

        print(" ¡Carga inmutable a la Capa Bronze completada con éxito!")
        registrar_log(proceso_nombre, "BRONZE", "EXITOSO", registros=total)

    except Exception as e:
        print(f" Error durante la ingesta a la Capa Bronze: {e}")
        registrar_log(proceso_nombre, "BRONZE", "ERROR", error=str(e))

if __name__ == "__main__":
    cargar_csv_a_bronze()