import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

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

def extraer_silver_paginado(tamano_pagina=1000):
    """Extrae todos los registros de silver.citas_cleaned paginando"""
    print(" Leyendo datos limpios desde silver.citas_cleaned...")
    registros = []
    inicio = 0
    while True:
        respuesta = supabase.schema("silver").table("citas_cleaned") \
            .select("*") \
            .range(inicio, inicio + tamano_pagina - 1) \
            .execute()
        
        datos = respuesta.data
        if not datos:
            break
        registros.extend(datos)
        inicio += tamano_pagina
        if len(datos) < tamano_pagina:
            break
    
    print(f" Se leyeron {len(registros)} registros de la Capa Silver.")
    return pd.DataFrame(registros)

def entrenar_y_poblar_gold():
    proceso_nombre = "TRAIN_MODEL_AND_GOLD_LOAD"
    registrar_log(proceso_nombre, "GOLD", "INICIADO")
    
    try:
        # 1. Extraer data de Silver
        df = extraer_silver_paginado()
        if df.empty:
            print(" No hay registros en Silver para entrenar.")
            return

        print(" Preparando dataset de entrenamiento para la Capa Gold...")

        # Variables predictoras (Features) y Target
        features = [
            'lead_time', 'age', 'scholarship', 'hipertension', 
            'diabetes', 'alcoholism', 'handcap', 'sms_received', 'historical_noshows'
        ]
        target = 'no_show_target'

        X = df[features]
        y = df[target]

        # 2. Poblar gold.ml_train_dataset en lotes
        print(" Poblando la tabla gold.ml_train_dataset en Supabase...")
        df_ml_train = df[['appointment_id'] + features + [target]].copy()
        registros_gold = df_ml_train.to_dict(orient='records')
        
        tamano_lote = 1000
        for i in range(0, len(registros_gold), tamano_lote):
            lote = registros_gold[i:i+tamano_lote]
            supabase.schema("gold").table("ml_train_dataset").insert(lote).execute()

        print(" Capa Gold (ml_train_dataset) poblada con éxito.")

        # 3. Dividir datos en Entrenamiento y Prueba (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(" Entrenando el modelo de Machine Learning (Random Forest)...")
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        # 4. Evaluar el modelo
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_proba)
        print(f" Evaluación del Modelo - ROC-AUC Score: {auc_score:.4f}")
        print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

        # 5. Guardar el modelo entrenado en la carpeta models/
        os.makedirs("models", exist_ok=True)
        ruta_modelo = "models/model.pkl"
        with open(ruta_modelo, "wb") as f:
            pickle.dump({"model": model, "features": features}, f)

        print(f" Modelo guardado exitosamente en '{ruta_modelo}'.")
        registrar_log(proceso_nombre, "GOLD", "EXITOSO", registros=len(df))

    except Exception as e:
        print(f" Error durante el entrenamiento y poblado Gold: {e}")
        registrar_log(proceso_nombre, "GOLD", "ERROR", error=str(e))

if __name__ == "__main__":
    entrenar_y_poblar_gold()