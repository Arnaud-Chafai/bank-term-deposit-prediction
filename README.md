## **Descripción del Proyecto**
Este repositorio contiene un proyecto de machine learning desarrollado para una **competencia de Kaggle**. El objetivo es predecir si un cliente bancario suscribirá un depósito a plazo utilizando un conjunto de datos de campañas de marketing anteriores. El proyecto aborda los siguientes puntos clave:
- Exploración y visualización de los datos para identificar patrones relevantes.
- Limpieza de datos y manejo de valores desconocidos en variables categóricas.
- Creación de nuevas características (feature engineering).
- Prueba de distintos modelos de clasificación.
- Optimización del mejor modelo mediante **Grid Search**.
- Evaluación de resultados con **curvas ROC y matrices de confusión**.

---

## **Dataset**
El dataset contiene información de clientes bancarios recopilada a partir de campañas de marketing anteriores.

- **Atributos macroeconómicos:** tasa de empleo trimestral, tasa Euribor a 3 meses, índice de precios al consumidor.
#### 🏦 **Datos del Cliente Bancario**
- **`age`**: Edad del cliente (numérico).  
- **`job`**: Tipo de trabajo (categórico).  
  - Valores: **admin.**, **blue-collar**, **entrepreneur**, **housemaid**, **management**, **retired**, **self-employed**, **services**, **student**, **technician**, **unemployed**, **unknown**.  
- **`marital_status`**: Estado civil (categórico).  
  - Valores: **divorced**, **married**, **single**, **unknown** (Nota: **divorced** incluye divorciados y viudos).  
- **`education_level`**: Nivel de educación (categórico).  
  - Valores: **basic.4y**, **basic.6y**, **basic.9y**, **high.school**, **illiterate**, **professional.course**, **university.degree**, **unknown**.  
- **`is_default`**: ¿Tiene crédito en impago? (categórico).  
  - Valores: **no**, **yes**, **unknown**.  
- **`housing_type`**: ¿Tiene préstamo hipotecario? (categórico).  
  - Valores: **no**, **yes**, **unknown**.  
- **`loan`**: ¿Tiene préstamo personal? (categórico).  
  - Valores: **no**, **yes**, **unknown**.  

#### 📞 **Datos del Último Contacto de la Campaña Actual**
- **`contact`**: Tipo de comunicación (categórico).  
  - Valores: **cellular**, **telephone**.  
- **`month`**: Mes del último contacto (categórico).  
  - Valores: **jan**, **feb**, **mar**, …, **nov**, **dec**.  
- **`day_of_week`**: Día de la semana del último contacto (categórico).  
  - Valores: **mon**, **tue**, **wed**, **thu**, **fri**.  

#### 🔄 **Otros Atributos**
- **`campaign`**: Número de contactos realizados durante esta campaña (numérico).  
- **`pdays`**: Días transcurridos desde el último contacto en una campaña anterior (numérico; **999** significa que no fue contactado previamente).  
- **`previous`**: Número de contactos previos a esta campaña (numérico).  
- **`poutcome`**: Resultado de la campaña de marketing anterior (categórico).  
  - Valores: **failure**, **nonexistent**, **success**.  

#### 📈 **Atributos del Contexto Económico y Social**
- **`emp_var_rate`**: Tasa de variación del empleo - indicador trimestral (numérico).  
- **`cons_price_index`**: Índice de precios al consumidor - indicador mensual (numérico).  
- **`cons_conf_index`**: Índice de confianza del consumidor - indicador mensual (numérico).  
- **`euribor_3m`**: Tasa Euribor a 3 meses - indicador diario (numérico).  
- **`n_employed`**: Número de empleados - indicador trimestral (numérico).  

---

### 🎯 **Variable de Salida (Objetivo)**
- **`output`**: ¿El cliente ha suscrito un depósito a plazo? (binario).  
  - Valores: **yes**, **no**.  
---

## **Flujo del Proyecto**
1. **Exploración de datos:**  
   - Visualización y análisis para comprender la distribución de las variables y la relación entre estas y la variable objetivo (`output`).
   - Se realizaron gráficos de barras y gráficos de dispersión para identificar patrones y correlaciones.
2. **Limpieza de datos:**  
   - Eliminación de valores irrelevantes.
   - Eliminación de duplicados y valores nulos.
3. **Feature Engineering:**  
   - Creación de nuevas características relevantes para mejorar el rendimiento del modelo.
   - Transformación de variables categóricas en variables binarias y combinación de variables clave.

---

## **Análisis de Clustering y Feature Engineering**

### **Clusters del Contexto Económico y Social + PCA** 📊  
Para enriquecer el análisis, se realizó una **reducción de dimensionalidad con PCA** sobre las variables económicas. Con los componentes principales obtenidos, se aplicó **K-means** para identificar patrones económicos.


<div align="center">
  <img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/Silhouette.png">
</div>

### 🟢 **Economía en Crecimiento Fuerte**  
- 📈 Alta **creación de empleo**  
- **Indicador principal:** `emp_var_rate` superior a la media  

### 🔴 **Economía en Recesión**  
- 📉 Muy baja **creación de empleo**  
- **Indicador principal:** `emp_var_rate` muy por debajo de la media  
<div align="center">
  <img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/pca.png">
</div>
---


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
## **Resultados**
- **Modelo ganador:** Random Forest con validación cruzada y optimización de hiperparámetros.
- **Puntuación final:**  
  - **AUC en validación:** 0.803  
  - **AUC en Kaggle:** 0.780
 <img src="https://github.com/Arnaud-Chafai/bank-term-deposit-prediction/blob/main/Screenshots/Curve%20ROC.png">
