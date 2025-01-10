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
Para enriquecer el análisis, se realizó un clustering de las variables económicas clave (`emp_var_rate` y `euribor_3m`) mediante **K-means**, seguido de una reducción de dimensionalidad con **PCA** para visualizar los grupos formados.

<div align="center">
  <img src="ruta_a_la_imagen_del_pca.png" alt="PCA por Clusters" width="600">
</div>

### 🟢 **Economía en Crecimiento Fuerte**  
- 📈 Alta **creación de empleo**  
- **Indicador principal:** `emp_var_rate` superior a la media  

### 🔴 **Economía en Recesión**  
- 📉 Muy baja **creación de empleo**  
- **Indicador principal:** `emp_var_rate` muy por debajo de la media  

---

### **Transformación de Características Categóricas**  
Se creó una nueva característica combinando estado civil y préstamo hipotecario (`status_marital_housing`) para identificar relaciones más complejas con la variable objetivo.

#### **Distribución del Estado Marital y Hipoteca por Output:**  
Este gráfico de barras muestra cómo las combinaciones de estado civil e hipoteca influyen en la decisión de suscribir un depósito a plazo.

<div align="center">
  <img src="ruta_a_la_imagen_del_barchart.png" alt="Gráfico de Barras de Feature Combinada" width="600">
</div>

---

### **Código para replicar el gráfico:**

```python
# Importar bibliotecas
import seaborn as sns
import matplotlib.pyplot as plt

# Crear una figura con tamaño específico
plt.figure(figsize=(10, 4))

# Crear el gráfico de conteo para la característica combinada
sns.countplot(
    x='status_marital_housing', 
    data=df, 
    hue='output', 
    linewidth=1.5, 
    edgecolor='black', 
    palette="coolwarm"
)

# Renombrar las etiquetas del eje X manualmente
plt.xticks(
    ticks=[0, 1, 2, 3], 
    labels=[
        "Solteros con hipoteca", 
        "Solteros sin hipoteca", 
        "En pareja con hipoteca", 
        "En pareja sin hipoteca"
    ]
)

# Configurar título y etiquetas de los ejes
plt.title('Distribución del Estado Marital y Hipoteca por Output')
plt.xlabel('Estado Marital y Hipoteca')
plt.ylabel('Cantidad')
plt.grid(axis="y", linestyle='--', color="black", alpha=0.3)
plt.gcf().autofmt_xdate()

# Mostrar el gráfico
plt.show()
```
## **Resultados**
- **Modelo ganador:** Random Forest con validación cruzada y optimización de hiperparámetros.
- **Puntuación final:**  
  - **AUC en validación:** 0.803  
  - **AUC en Kaggle:** 0.780  

