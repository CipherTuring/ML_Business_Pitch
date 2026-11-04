# Ancla: retención de clientes para ISP regionales

Solución de Machine Learning sobre el dataset **Telco Customer Churn**. Predice qué clientes pueden abandonar, explica por qué y ayuda a repartir un presupuesto de retención limitado.

Proyecto individual de la asignatura Aprendizaje Automático.

## Contenido

| Ruta | Descripción |
|---|---|
| `churn_ml.ipynb` | Datos, modelos, optimización, Pareto, SHAP, LIME y conclusiones |
| `streamlit_app.py` | Aplicación interactiva con cuatro secciones |
| `src/churn_common.py` | Limpieza, esquema y cálculos compartidos |
| `artifacts/` | Modelos exportados y resultados que carga la app |
| `docs/propuesta_negocio.md` | Propuesta comercial |
| `docs/guia_defensa.md` | Guía para la negociación |
| `telco_churn.ipynb` | Plantilla de clase usada como referencia |

## Ejecutar en local

Requiere Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux o macOS
pip install -r requirements.txt
streamlit run streamlit_app.py
```

La app solo carga los modelos de `artifacts/`; no reentrena. Una sección puede abrirse directamente con `?seccion=1` a `?seccion=4`.

## Reproducir el notebook

Además de `requirements.txt`, instala `jupyter`, abre `churn_ml.ipynb` y ejecuta todas las celdas desde la raíz del proyecto.

Las búsquedas de hiperparámetros se guardan en `artifacts/search/`, porque la latencia medida varía entre corridas y puede alterar el frente de Pareto. Con `FORCE_RERUN_SEARCH = False` se reproducen las cifras citadas en el texto. Con `True` se repiten las búsquedas desde cero; en ese caso los números pueden cambiar un poco y el texto dejaría de coincidir.

## Publicar en Streamlit Community Cloud

1. Sube el proyecto a un repositorio público de GitHub. `.gitignore` ya excluye `.venv/`.
2. Entra en [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
3. Elige **Create app** y luego el repositorio y la rama.
4. En **Main file path** escribe `streamlit_app.py`.
5. En **Advanced settings** elige **Python 3.12**. Debe coincidir: los modelos se guardaron con estas versiones de `requirements.txt`.
6. Pulsa **Deploy** y comprueba las cuatro secciones.

La app de la nube se suspende tras un tiempo sin visitas. Ábrela unos minutos antes de la presentación.

## Metodología

- Partición estratificada 80/20 con `random_state=42`. El conjunto de test solo se usa tras cerrar la selección.
- Preprocesamiento dentro de `Pipeline` y `ColumnTransformer`, para evitar fuga de información.
- Validación cruzada estratificada de 5 folds, con las mismas particiones en todo el proyecto.
- Grid Search, Random Search, TPE y NSGA-II con 54 configuraciones cada uno.
- Multiobjetivo: maximizar F1 y minimizar la latencia de inferencia.
- Explicaciones con SHAP y LIME para la clase Churn = Yes.

## Limitaciones

- El dataset describe una compañía ficticia de EE. UU. Para usar el modelo en un ISP real hay que reentrenarlo con sus datos.
- Los valores económicos de la app son supuestos hipotéticos, no resultados del modelo.
- Las explicaciones indican asociación, no causalidad. Una acción de retención solo se valida con un piloto y grupo de control.
