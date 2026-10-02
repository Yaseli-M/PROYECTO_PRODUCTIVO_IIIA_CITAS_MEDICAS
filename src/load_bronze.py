import os
import sys
import pandas as pd

# Asegurar que Python reconozca los módulos dentro de src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import supabase

def cargar_csv_a_bronze(ruta_csv="Data/KaggleV2-May-2016.csv"):
    print(" Leyendo archivo CSV desde la carpeta Data/...")
    df = pd.read_csv(ruta_csv)

    # Renombrar columnas para que coincidan con la tabla en Supabase (minúsculas)
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

    registros = df.to_dict(orient='records')
    tamano_lote = 1000
    
    print(f" Cargando {len(registros)} registros en la capa Bronze (esquema bronze)...")
    for i in range(0, len(registros), tamano_lote):
        lote = registros[i:i+tamano_lote]
        supabase.schema("bronze").table("citas_raw").insert(lote).execute()
        print(f" Lote {i // tamano_lote + 1} insertado ({min(i + tamano_lote, len(registros))}/{len(registros)})")

    print(" ¡Carga completa exitosa a la capa Bronze!")

if __name__ == "__main__":
    cargar_csv_a_bronze()