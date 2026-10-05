import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score

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

def extraer_silver_paginado(tamano_pagina=1000):
    """Extrae registros de silver.citas_cleaned con paginación determinista"""
    print(" Leyendo datos limpios desde silver.citas_cleaned...")
    registros = []
    inicio = 0
    while True:
        respuesta = supabase.schema("silver").table("citas_cleaned") \
            .select("*") \
            .order("appointment_id", desc=False) \
            .range(inicio, inicio + tamano_pagina - 1) \
            .execute()
        datos = respuesta.data
        if not datos:
            break
        registros.extend(datos)
        inicio += tamano_pagina
        if len(datos) < tamano_pagina:
            break
    print(f" Total leídos de Silver: {len(registros)} registros.")
    return pd.DataFrame(registros)

def entrenar_y_comparar_modelos():
    proceso_nombre = "TRAIN_AND_BENCHMARK_GOLD_MODELS"
    registrar_log(proceso_nombre, "GOLD", "INICIADO")
    
    try:
        df = extraer_silver_paginado()
        if df.empty:
            print(" No hay registros en Silver para entrenar.")
            return

        features = [
            'lead_time', 'age', 'scholarship', 'hipertension', 
            'diabetes', 'alcoholism', 'handcap', 'sms_received', 'historical_noshows'
        ]
        target = 'no_show_target'

        X = df[features]
        y = df[target]

        # 1. Poblar la Capa Gold (gold.ml_train_dataset) en Supabase
        print(" Poblando la tabla gold.ml_train_dataset en Supabase...")
        df_ml_train = df[['appointment_id'] + features + [target]].copy()
        df_ml_train = df_ml_train.where(pd.notnull(df_ml_train), None)
        registros_gold = df_ml_train.to_dict(orient='records')
        
        tamano_lote = 1000
        total_gold = len(registros_gold)
        for i in range(0, total_gold, tamano_lote):
            lote = registros_gold[i:i + tamano_lote]
            supabase.schema("gold").table("ml_train_dataset").insert(lote).execute()

        print(" Capa Gold (ml_train_dataset) poblada exitosamente.")

        # 2. Split Estratificado (80% Train / 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Ratio exacto de desbalance para XGBoost (Negativos / Positivos)
        ratio_desbalance = (len(y_train) - sum(y_train)) / sum(y_train)

        # 3. Definición de los 4 Modelos ajustados para Desbalance de Clases
        modelos = {
            "LogisticRegression_Baseline": LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, max_depth=6, scale_pos_weight=ratio_desbalance, 
                eval_metric='logloss', random_state=42
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=100, class_weight='balanced', random_state=42, verbose=-1
            )
        }

        mejor_modelo_nombre = None
        mejor_recall = 0.0
        objeto_mejor_modelo = None

        print("\n===  INICIANDO BENCHMARKING DE MODELOS ===")
        os.makedirs("models", exist_ok=True)

        for nombre, modelo in modelos.items():
            print(f"\n Entrenando {nombre}...")
            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]

            rec = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            auc = roc_auc_score(y_test, y_proba)

            print(f"   ➜ Recall (Faltó): {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")

            # Guardar artefacto individual de cada modelo
            ruta_individual = f"models/{nombre.lower()}_v1.pkl"
            with open(ruta_individual, "wb") as f:
                pickle.dump({"model": modelo, "features": features}, f)

            # Criterio de Selección del Campeón: Mayor Recall en pacientes que faltan (clase 1)
            if rec > mejor_recall:
                mejor_recall = rec
                mejor_modelo_nombre = nombre
                objeto_mejor_modelo = modelo

        # 4. Guardar el Modelo Campeón
        print(f"\n ¡MODELO CAMPEÓN SELECCIONADO!: {mejor_modelo_nombre} (Recall: {mejor_recall:.4f})")
        ruta_campeon = "models/champion_model.pkl"
        with open(ruta_campeon, "wb") as f:
            pickle.dump({
                "model": objeto_mejor_modelo, 
                "features": features, 
                "model_name": mejor_modelo_nombre,
                "recall": mejor_recall
            }, f)

        print(f" Modelo Campeón guardado exitosamente en '{ruta_campeon}'.")
        registrar_log(proceso_nombre, "GOLD", "EXITOSO", registros=total_gold)

    except Exception as e:
        print(f" Error durante entrenamiento Gold: {e}")
        registrar_log(proceso_nombre, "GOLD", "ERROR", error=str(e))

if __name__ == "__main__":
    entrenar_y_comparar_modelos()