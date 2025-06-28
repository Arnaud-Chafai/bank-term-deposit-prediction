### Bank Term Deposit Prediction: Kaggle Challenge
<img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/BANK.png">

## **Descripción del Proyecto**
Este repositorio contiene un proyecto de machine learning desarrollado para una **competencia de Kaggle**. El objetivo es predecir si un cliente bancario suscribirá un depósito a plazo utilizando un conjunto de datos de campañas de marketing anteriores.
Cabe destacar que la **métrica de evaluación utilizada es el AUC **, ya que es la métrica de referencia establecida por la competencia para medir el desempeño de los modelos participantes.

El proyecto aborda los siguientes puntos clave:
- **Exploración y visualización de los datos** para identificar patrones relevantes.
- **Limpieza de datos** y manejo de valores desconocidos.
- **Análisis exploratorio de los datos** para entender mejor la distribución de las variables y sus correlaciones.
- **Creación de nuevas características (feature engineering)** en base a las **variables categóricas**, que representan una proporción importante en el dataset.
- **Clustering y reducción de dimensionalidad:**  
  - Se aplicó **PCA** para reducir la cantidad de variables correlacionadas.
  - Se realizaron **clusterings** para agrupar observaciones basadas en el contexto socioeconómico.
- **Selección de características (feature selection)** para mejorar el rendimiento de los modelos.
- **Prueba de distintos modelos de clasificación.**
- **Optimización de los modelos mediante Grid Search** y ajuste de hiperparámetros.
- **Evaluación de los resultados** utilizando la **métrica AUC (Área Bajo la Curva ROC)** como referencia principal para medir el desempeño de los modelos.
---

## **Ejemplo del Flujo de Trabajo para el Análisis Socioeconómico**  
Este apartado muestra el flujo de trabajo seguido para el análisis de las variables socioeconómicas y cómo se trataron para mejorar el rendimiento de los modelos de machine learning.

### **Análisis de Correlación y Reducción de Dimensionalidad (PCA)**  
Durante la exploración de los datos, se observó que varias variables socioeconómicas presentaban una **alta correlación** entre sí. Esto puede generar redundancia y afectar el rendimiento del modelo, por lo que se decidió realizar una **reducción de dimensionalidad**.

### **Aplicación de PCA (Análisis de Componentes Principales)**  
Se aplicó **PCA** (Análisis de Componentes Principales) para reducir la cantidad de variables manteniendo la mayor cantidad posible de información relevante.

- **Explicación:** Se conservaron **tres componentes principales** que explican el **99.3% de la variabilidad** de los datos.
- El número de componentes se seleccionó utilizando el **método del codo**, como se muestra en la siguiente gráfica:

<img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/codo.png" alt="Método del Codo" width="600">

---

### **Clusters del Contexto Económico y Social + PCA** 📊  
Con los **tres componentes principales** obtenidos, se aplicó **K-means** para identificar patrones económicos y agrupar a los clientes en diferentes clusters según su contexto socioeconómico.

<img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/Silhouette.png" alt="Silhouette Score" width="600">

---

### **Descripción de los Clusters Económicos:**  

### 🟢 **Economía en Crecimiento Fuerte**  
- 📈 Alta **creación de empleo**  
- **Indicador principal:** `emp_var_rate` superior a la media  

### 🔴 **Economía en Recesión**  
- 📉 Muy baja **creación de empleo**  
- **Indicador principal:** `emp_var_rate` muy por debajo de la media  

---

### **Visualización de los Clusters en el Espacio PCA:**  
Finalmente, los clusters formados fueron proyectados en el espacio definido por los **componentes principales**.

<img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/pca.png" alt="Visualización PCA Clusters" width="600">

---

### **Resumen:**  
- Se realizó un análisis de correlación para identificar variables que no aportan valor al modelo.
- Se aplicó **PCA** para reducir las dimensiones y mantener el **99.3%** de la información.
- Utilizando los componentes principales, se ejecutó **K-means** para identificar **clusters económicos**.
- Los resultados evidencian dos grupos principales:  
  1. **Economía en Crecimiento Fuerte**  
  2. **Economía en Recesión**

## **Resultados**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# ===== Parámetros para GridSearch =====
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [7, 9, 11],
    'min_samples_leaf': [3, 5, 7],
    'min_samples_split': [25, 35, 45],
    'max_features': ['log2', 'sqrt'],
    'max_samples': [0.8, 0.9, 0.95],
}

# ===== Crear el modelo =====
rf_model = RandomForestRegressor(random_state=42)

# ===== Definir el GridSearchCV =====
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),  # MSE negativo para minimizar
    cv=5,                   # Validación cruzada de 5 folds
    n_jobs=-1,              # Usar todos los núcleos disponibles
    verbose=2               # Mostrar progreso detallado
)

# ===== Ejecutar GridSearch =====
grid_search.fit(X_train, y_train)

# ===== Mostrar los mejores parámetros =====
print("\n🔍 Mejores parámetros encontrados:")
print(grid_search.best_params_)

# ===== Mejor Score (RMSE) =====
best_rmse = np.sqrt(-grid_search.best_score_)
print(f"\nMejor RMSE obtenido: {best_rmse:.4f}")

# ===== Crear el modelo final con los mejores parámetros =====
best_rf_model = grid_search.best_estimator_

# ===== Evaluar el modelo en el conjunto de prueba =====
y_pred = best_rf_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE en el conjunto de prueba: {test_rmse:.4f}")

```



- **Modelo con mejor rendimiento:** Random Forest con validación cruzada y optimización de hiperparámetros.
- **Puntuación final:**  
  - **AUC en validación:** 0.803  
  - **AUC en Kaggle:** 0.780
 <img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/Curve%20ROC.png">
