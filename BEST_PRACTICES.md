---

*(Este archivo documenta las buenas prácticas de gobernanza, MLOps y código que exigió el docente durante el ciclo).*

```markdown
# 🛡️ Buenas Prácticas de Gobernanza, MLOps y Código - MediPredict AI

En cumplimiento con los estándares técnicos exigidos en la asignatura **Proyecto Productivo IIIA**, el proyecto implementa los siguientes principios:

---

## 1. 🔐 Seguridad y Gobernanza de Datos
- **Protección de Credenciales:** Las llaves privadas y credenciales de la API de Supabase se gestionan exclusivamente a través de variables de entorno `.env` ignoradas en Git mediante `.gitignore`.
- **Plantilla Pública:** Se suministra el archivo `.env.example` con la estructura requerida sin exponer secrets del proyecto.

## 2. 🔄 Reproducibilidad y Control de Experimentos
- **Fijación de Semilla Global (`seed = 42`):** Todos los procesos estocásticos (partición de datos `train_test_split`, inicialización de Random Forest, XGBoost y la Simulación de Montecarlo) utilizan la constante `42` para garantizar resultados idénticos en cualquier entorno.

## 3. ⚖️ Manejo Ético del Desbalance de Clases
- **Priorización del Recall sobre el Accuracy:** Dado el desbalance de clases (20.19% inasistencia), se descartó el uso de Accuracy por ser una métrica engañosa. Se optó por ajustar los pesos de clase (`class_weight='balanced'`) y realizar *Threshold Tuning* (umbral 0.48) para maximizar la detección de pacientes en riesgo.

## 4. 🗄️ Inmutabilidad y Auditoría en la Nube
- **Arquitectura Medallion en Supabase:** La capa `bronze.citas_raw` mantiene el registro original e inalterable de ingesta.
- **Trazabilidad de Procesos:** Cada ejecución de los scripts (`01_ingest_bronze.py`, `02_process_silver.py`, etc.) registra automáticamente su timestamp de inicio, fin, cantidad de filas procesadas y estado en la tabla `meta.log_procesos`.

## 5. 🧑‍⚕️ Supervisión Humana (Human-in-the-Loop)
- **Gobernanza Ética en Salud:** Las decisiones del modelo no se aplican de forma autónoma sobre diagnósticos o cancelaciones de citas. El aplicativo en Streamlit integra un módulo interactivo para que el analista de admisión audite las alertas y registre su dictamen en `gold.feedback_loop`.
