# Despliegue de la Predicción de Inversión en Videojuegos

Este repositorio contiene el código y los recursos necesarios para el despliegue de un modelo de clasificación que predice la inversión en videojuegos. El despliegue se realiza a través de una interfaz de usuario interactiva construida con Streamlit.

## 🎮 Predicción de Inversión – Videojuegos

### Descripción del Proyecto
Este proyecto implementa un modelo de clasificación (Red Neuronal) para predecir la inversión en videojuegos basándose en características del jugador como edad, sexo, plataforma preferida y si es un consumidor habitual, así como el videojuego de interés. La predicción se presenta a través de una aplicación web interactiva.

### Estructura del Repositorio
- `modelo-NN.pkl`: Archivo binario que contiene el modelo de Red Neuronal entrenado, el escalador `MinMaxScaler` y la lista de variables (`variables`) utilizadas durante el entrenamiento.
- `videojuegos-datosFuturos.csv` (opcional): Archivo CSV con datos de ejemplo para pruebas futuras.
- Notebook de Colab (este documento): Contiene el código para cargar el modelo, preparar los datos y realizar predicciones, incluyendo la definición de la interfaz de Streamlit.

### Requisitos
Para ejecutar la aplicación y el modelo, necesitarás las siguientes librerías de Python:

- `numpy`
- `pandas`
- `matplotlib` (para visualizaciones, aunque no directamente en el despliegue de Streamlit)
- `scikit-learn` (para el `MinMaxScaler`)
- `streamlit` (para la interfaz de usuario web)

Puedes instalar estas dependencias usando `pip`:
```bash
pip install numpy pandas matplotlib scikit-learn streamlit
```

### Pasos para el Despliegue

#### 1. Cargar el Modelo
El modelo entrenado, el escalador de características y las variables utilizadas para el entrenamiento se cargan desde el archivo `modelo-NN.pkl`.

```python
import pickle
filename = 'modelo-NN.pkl'
modelo, min_max_scaler, variables = pickle.load(open(filename, 'rb'))
```

#### 2. Cargar y Preparar Datos Futuros
La aplicación Streamlit permite al usuario introducir datos en tiempo real. Estos datos se procesan de la siguiente manera:

- **Captura de Datos**: La interfaz de Streamlit captura la edad, sexo, plataforma y si el usuario es un consumidor habitual, así como el videojuego seleccionado.
- **Creación de DataFrame**: Los datos introducidos se convierten en un DataFrame de pandas.
- **Codificación One-Hot**: Las variables categóricas (`videojuego`, `Plataforma`, `Sexo`, `Consumidor_habitual`) se transforman usando codificación one-hot (`pd.get_dummies`).
- **Ajuste de Columnas**: Se aseguran de que todas las columnas necesarias para el modelo estén presentes, reindexando el DataFrame y llenando los valores faltantes con cero.
- **Normalización de 'Edad'**: La columna `Edad` se normaliza utilizando el `MinMaxScaler` previamente guardado con el modelo.

#### 3. Realizar la Predicción
Una vez que los datos están preparados, el modelo cargado se utiliza para hacer una predicción.

```python
Y_pred = modelo.predict(data_preparada)
```
La predicción (`Y_pred`) se añade como una nueva columna al DataFrame original de los datos de entrada para facilitar la visualización.

#### 4. Ejecutar la Aplicación Streamlit

El archivo de Streamlit (`f-XAsW3kViMO` en este notebook) contiene la lógica para la interfaz de usuario. Para ejecutar la aplicación:

1.  **Guarda el código de la celda `f-XAsW3kViMO`** en un archivo llamado, por ejemplo, `app.py`.
2.  **Asegúrate de que `modelo-NN.pkl`** esté en el mismo directorio o accesible por la aplicación.
3.  **Abre tu terminal** o Anaconda Prompt y navega hasta el directorio donde guardaste `app.py`.
4.  **Ejecuta el siguiente comando**:
    ```bash
    streamlit run app.py
    ```

Esto abrirá la aplicación en tu navegador web, donde podrás interactuar con ella para obtener predicciones.

