# **bank-term-deposit-prediction**

**Proyecto de machine learning para la competencia de Kaggle sobre predicción de depósitos a plazo bancarios.** Se realiza una exploración y limpieza de datos, junto con la creación de características. Se prueban distintos modelos de clasificación como Regresión Logística, KNN, XGBoost y Random Forest. Finalmente, el modelo seleccionado fue un **Random Forest optimizado mediante Grid Search**, que obtuvo un **AUC de 0.803** en validación y **0.780** en la competencia.

---

## **Índice**
1. [Descripción del Proyecto](#descripción-del-proyecto)  
2. [Dataset](#dataset)  
3. [Flujo del Proyecto](#flujo-del-proyecto)  
4. [Resultados](#resultados)  
5. [Análisis de Clustering y Feature Engineering](#análisis-de-clustering-y-feature-engineering)  
6. [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)  
7. [Conclusión](#conclusión)  

---

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
El dataset contiene información de clientes bancarios recopilada a partir de campañas de marketing anteriores. Algunas de las variables incluidas son:
- **Datos del cliente:** edad, estado civil, nivel educativo, entre otros.
- **Información del último contacto de la campaña:** tipo de comunicación, día y mes del contacto.
- **Atributos macroeconómicos:** tasa de empleo trimestral, tasa Euribor a 3 meses, índice de precios al consumidor.

---

## **Flujo del Proyecto**
1. **Exploración de datos:**  
   Visualización y análisis para comprender la distribución de las variables y la relación entre estas y la variable objetivo (`output`).
2. **Limpieza de datos:**  
   - Eliminación de valores irrelevantes y tratamiento de valores "unknown".
3. **Feature Engineering:**  
   - Creación de nuevas características relevantes para mejorar el rendimiento del modelo.

---

## **Análisis de Clustering y Feature Engineering**

### **Clusters del Contexto Económico y Social + PCA** 📊
Para enriquecer el análisis, se realizó un clustering de las variables económicas clave (`emp_var_rate` y `euribor_3m`) mediante K-means, seguido de una reducción de dimensionalidad con PCA para visualizar los grupos formados.

<div align="center">
  <img src="ruta_a_la_imagen_del_pca.png" alt="PCA por Clusters" width="600">
</div>

### 🟢 **Cluster 0: Economía en Crecimiento Fuerte**  
- **Razón:**  
  - 📈 Alta **creación de empleo**  
- **Indicadores Clave:**  
  - 🔹 `emp_var_rate`: Superior a la media en **+1298.6%**  
  - 🔹 `euribor_3m`: Superior a la media en **+33.1%**  

### 🔴 **Cluster 1: Economía en Recesión**  
- **Razón:**  
  - 📉 Muy baja **creación de empleo**  
- **Indicadores Clave:**  
  - 🔹 `emp_var_rate`: Inferior a la media en **-2656.4%**  
  - 🔹 `euribor_3m`: Inferior a la media en **-67.7%**

---

### **Transformación de Características Categóricas**
Se creó una nueva característica combinando estado civil y préstamo hipotecario (`status_marital_housing`) para identificar relaciones más complejas con la variable objetivo.

#### **Distribución del Estado Marital y Hipoteca por Output:**
Este gráfico de barras muestra cómo las combinaciones de estado civil e hipoteca influyen en la decisión de suscribir un depósito a plazo.

<div align="center">
  <img src="ruta_a_la_imagen_del_barchart.png" alt="Gráfico de Barras de Feature Combinada" width="600">
</div>

---

### **Silhouette Score para Validación de Clustering**
Se utilizó el Silhouette Score para validar la calidad de los clusters formados y asegurar que la agrupación refleja correctamente las diferencias en el contexto económico-social.

<div align="center">
  <img src="ruta_a_la_imagen_del_silhouette_score.png" alt="Silhouette Score" width="600">
</div>

---

## **Resultados**
- **Modelo ganador:** Random Forest con validación cruzada y optimización de hiperparámetros.
- **Puntuación final:**  
  - **AUC en validación:** 0.803  
  - **AUC en Kaggle:** 0.780  

---

## **Cómo Ejecutar el Proyecto**
1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/bank-term-deposit-prediction.git
