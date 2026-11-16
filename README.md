# 🩺 MediPredict AI - Sistema Inteligente de Gestión Preventiva de Citas Médicas

> **Proyecto Productivo IIIA** | Instituto Continental (Huancayo, Perú - 2026)  
> **Docente:** Sergio Víctor Orizano Salvador  
> **Carrera:** Ciencia de Datos e Inteligencia Artificial (5to Ciclo)

---

## 👥 Equipo de Desarrollo
- **Alarcón Olmedo, Víctor Efraín**
- **Inuma Macedo, Mariana Milagros**
- **Mendez Macedo, Karen Yaseli**

---

## 📌 Descripción del Proyecto
**MediPredict AI** es una solución MLOps basada en la arquitectura **Lakehouse Medallion Cloud en Supabase (PostgreSQL)** y **Streamlit**. El sistema predice con días de anticipación la probabilidad de inasistencia ("No-Show") de los pacientes a citas médicas ambulatorias, recomendando acciones preventivas personalizadas (llamadas prioritarias, notificaciones por WhatsApp o correos estándar) e integrando un módulo de **Validación Humana (*Human-in-the-Loop*)** para auditoría.

---

## 🏗️ Arquitectura de Datos Medallion (11 Tablas en Supabase)
- **Bronze (`bronze.citas_raw`):** Ingesta cruda e inmutable de 110,527 citas.
- **Silver (`silver.citas_cleaned`):** Limpieza, deduplicación y cálculo de `lead_time` e `historical_noshows`.
- **Gold - Modelo Estrella:** `gold.dim_paciente`, `gold.dim_barrio`, `gold.dim_tiempo`, `gold.fact_citas`.
- **Gold - MLOps:** `gold.ml_train_dataset`, `gold.score_input`, `gold.score_output`, `gold.feedback_loop`.
- **Meta:** `meta.log_procesos` (Trazabilidad e inmutabilidad de ejecuciones ETL).

---

## 📊 Modelo Campeón y Métricas
- **Algoritmo Campeón:** Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `class_weight='balanced'`)
- **Sensibilidad (Recall):** **0.8600 (86.00%)**
- **ROC-AUC:** **0.7352**
- **Umbral de Decisión (Threshold Tuning):** **0.48**
- **Falsos Negativos:** Solo 625 citas no detectadas de 4,464 ausentes en el conjunto de prueba.
- **Impacto Económico (Montecarlo 10k iteraciones):** **ROI 11 : 1** (Ahorro neto est. S/ 17,332/mes).

---

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Yaseli-M/PROYECTO_PRODUCTIVO_IIIA_CITAS_MEDICAS.git
   cd PROYECTO_PRODUCTIVO_IIIA_CITAS_MEDICAS