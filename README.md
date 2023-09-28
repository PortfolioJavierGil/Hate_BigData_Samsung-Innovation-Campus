El presente texto proporciona una visión general de un proyecto centrado en la obtención de comentarios de redes sociales y el desarrollo de un modelo de inteligencia artificial para **detectar el discurso de odio**. A lo largo de esta introducción, exploraremos los enfoques utilizados para obtener comentarios de plataformas como YouTube, YouTube Live, TikTok, Reddit e Instagram, así como los modelos desarrollados para clasificar los comentarios como comentarios de odio o comentarios no ofensivos.

* En primer lugar, se describe el enfoque adoptado para obtener los datos de cada plataforma.
* Tras esto, se procesan los datos utilizando Apache Spark.
* Posteriormente, se explora el desarrollo de un modelo de inteligencia artificial para detectar el discurso de odio. Se presentan dos enfoques: uno implementado en Pyspark y otro en Python. Antes de construir el modelo, se realiza un análisis de los datos de entrenamiento, calculando características de texto y visualizando el equilibrio de los datos. En el modelo de Pyspark, se muestra el proceso de construcción del modelo de clasificación de texto utilizando Spark, mientras que en el modelo de Python se entrenan modelos como XGBoost y una red neuronal con Keras (LSTM).
* Finalmente, se introduce el código que etiqueta los comentarios recopilados de las redes sociales de Samsung utilizando el modelo de inteligencia artificial previamente desarrollado. Este proceso de etiquetado permite clasificar los comentarios como comentarios de odio o comentarios no ofensivos, lo cual resulta útil para comprender y analizar el contenido de los comentarios en las redes sociales.
* En resumen, este proyecto aborda la obtención de comentarios de redes sociales, el desarrollo de un modelo de inteligencia artificial para detectar el discurso de odio y el etiquetado de comentarios en redes sociales específicas. A través de estos enfoques, se busca obtener información relevante y procesable a partir de los comentarios en las plataformas digitales más populares.

<hr>

# Obtención de Comentarios de Redes Sociales
[Archivos de jupyter explicativos](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/GetDataFromSM)
<br>
[Datos extraidos](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/Data)

Para llevar a cabo el objetivo de obtener comentarios de las redes sociales YouTube, TikTok, Reddit e Instagram, para videos de un canal específico como para publicaciones generales. Se ha utilizado el siguiente enfoque según la plataforma:

1. [**YouTube**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/YouTubeAPI.ipynb): Se utiliza la API de YouTube para acceder a los comentarios de los videos de un canal específico. Se pasa el link del canal como argumento y se obtienen todas las estadísticas del canal y de sus videos, incluyendo los comentarios.

2. [**YouTube** **live**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/YouTubeLiveAPI.ipynb): Se utiliza la API de YouTube para acceder al chat de un video en directo. Se pasa el link del directo como argumento y se obtienen todosclos comentarios del chat.

3. [**TikTok**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/tiktokScrapping.ipynb): Se utiliza la técnica de web scraping para obtener los comentarios de las publicaciones de TikTok. Esto implica utilizar bibliotecas como BeautifulSoup o Selenium para extraer los comentarios directamente del HTML de la página.

4. [**Reddit**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/RedditAPI.ipynb): Se utiliza la API de Reddit para acceder a los comentarios de las comunidades y los posts. Permite especificar la comunidad o el post en particular del que deseas obtener comentarios y utilizar la API para obtener los datos correspondientes.

5. [**Instagram**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/InstagramScrapping.ipynb): Al igual que en el caso de TikTok, se utiliza web scraping para obtener los comentarios de las publicaciones de Instagram. Se utilizan bibliotecas como BeautifulSoup o Selenium para extraer los comentarios del HTML de la página.

6. [**4chan**](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/GetDataFromSM/4chanAPI.ipynb): Se utiliza la función 'get_all_threads()' que realiza una solicitud HTTP a la URL "https://a.4cdn.org/tablero/catalog.json" para obtener un archivo JSON que contiene información sobre hilos de discusión en el tablero de 4chan.

<hr>

# Procesamiento de los datos con Spark
[Archivo de jupyter explicativo](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/Data_Process/train_data_process.ipynb)
Los datos se procesan utilizando Spark y se unifican en un solo dataframe para poder facilitar el entrenamiento

<hr>

# Modelo de IA para detectar el hate
[Archivos de jupyter explicativos](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/Model)

Se han desarrollado dos modelos, uno en pyspark y otro en python.

<hr>

## Analisis de los datos de entrenamiento
Antes de elaborar el modelo se lleva a cabo el analisis de los datos de entrenamiento.

[Archivo de jupyter explicativo](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/Analysis/train_data_analysis.ipynb)

* Se calculan algunas características de texto, como la cuenta de palabras, la cuenta de palabras únicas, la cuenta de stop words, la longitud promedio de las palabras, la cuenta de caracteres y la cuenta de puntuación. Estas características se utilizan para el análisis de los datos para ver si pueden ser relevantes en la detección de hate speech.
* Se visualizan los datos del balanceo de los datos de train.
* Se generan y cuentan los unigramas, bigramas y trigramas. Los n-gramas son secuencias contiguas de n elementos en un texto, generalmente palabras. En el contexto del procesamiento de lenguaje natural, los n-gramas se utilizan para capturar la estructura y el significado de un texto al analizar la frecuencia y la co-ocurrencia de las palabras.


<hr>

## Pyspark
Análogamente se ha desarrollado un modelo en pyspark para la implementación de una situación real en el campo de BigData.

[Este código](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/Model/pyspark/pysparkML.ipynb) demuestra el proceso de construcción de un modelo de clasificación de texto utilizando Spark. Realiza los siguientes pasos:

* Importa los datos y realiza una limpieza del texto de los tweets.
* Divide los datos en conjuntos de entrenamiento y prueba.
* Realiza la limpieza de texto en los datos, incluyendo la eliminación de caracteres especiales y la normalización del texto.
* Tokeniza el texto y elimina las palabras vacías.
* Convierte el texto tokenizado en una matriz de recuentos de tokens.
* Entrena y evalúa modelos de clasificación: Naive Bayes, Árbol de decisión y GBT Classifier.
* Utiliza el modelo GBT Classifier para hacer predicciones en el conjunto de prueba.

<hr>

## Python
El modelo en python se ha llevado a cabo para aumentar el rendimiento frente al modelo de pyspark.

[Este código](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/Model/modeloPyhton.ipynb) realiza las siguientes tareas:

* Importa los datos.
* Realiza la limpieza de texto en los datos, incluyendo la eliminación de caracteres especiales y la normalización del texto.
* Utiliza embeddings pre-entrenados (GloVe y FastText) para obtener representaciones vectoriales de las palabras.
* Divide los datos en conjuntos de entrenamiento y prueba.
* Utiliza el CountVectorizer para convertir el texto en una matriz de recuentos de tokens.
* Entrena y evalúa modelos de clasificación: XGBoost y una red neuronal con Keras (LSTM).
* Realiza predicciones en un conjunto de datos de prueba utilizando el modelo de keras entrenado.
* Guarda el modelo y el tokenizer para su uso posterior.

### Red Neuronal (LSTM)
Cabe destacar la arquitectura de la red neuronal empleada en el modelo usado para la predicción

* **Capa de Embedding (Incrustación)**: Esta capa convierte las palabras de entrada en vectores densos de longitud fija. En este caso, se utilizan vectores de longitud 100 para representar las palabras. La capa tiene 5,000,000 de parámetros (pesos) para aprender a generar los vectores de palabras adecuados.
* **Capa de Dropout Espacial**: Esta capa aplica el dropout para prevenir el sobreajuste en los datos. El dropout elimina aleatoriamente conexiones entre neuronas durante el entrenamiento para evitar que la red dependa demasiado de ciertas características.
* **Capa LSTM**: Esta capa utiliza unidades LSTM (Long Short-Term Memory) para modelar la secuencia de palabras en el texto. Las unidades LSTM tienen 100 celdas de memoria y pueden capturar patrones a largo plazo en datos secuenciales.
* **Capa Densa**: Esta es la capa de salida de la red, consiste en una sola neurona con una función de activación sigmoide. Genera una salida entre 0 y 1, que representa la probabilidad de que la instancia de texto pertenezca a una clase específica.


<img width="494" alt="image" src="https://github.com/aleo-phd/MOD-Samsung/assets/91088022/b78f472a-072d-485d-a664-74432d678d8b">

<hr>

# (Predicción) Detector Hate en redes sociales
[Archivos de jupyter explicativos](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/Prediction)

[Este código](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/blob/main/Prediction/generate_predictions_samsung_comments.ipynb) sirve para etiquetar comentarios de las redes sociales de Samsung. Se etiquetan los comentarios recopilados anteriormente y se utiliza el modelo de inteligencia artificial para clasificarlos como comentarios de hate o comentarios no hate.

<hr>

# Visualizacion en PowerBI

En este [link](https://app.powerbi.com/view?r=eyJrIjoiNDM5M2I5OWItZTUyNC00NzYzLTljNjAtZGYwMDQ0MTkzNzU2IiwidCI6IjZhZmVhODVkLWMzMjMtNDI3MC1iNjlkLWE0ZmIzOTI3YzI1NCIsImMiOjl9) se incluye la visualización de los datos extraidos y etiquetados usando el modelo explicado anteriormente.

## Visualizacion en Tiempo Real
[Archivo de jupyter expliactivo](https://github.com/aleo-phd/MOD-Samsung/blob/main/GetDataFromSM/powerBIPanel.ipynb)

En este codigo se envian en tiempo real los datos extraidos del chat de youtube para su representación en directo en powerBI.
Parte de esta representación se implementa en una [web](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/tree/main/web) para poder seguir el directo a la vez que se monitoriza la cantidad de hate de este.

Este es el diseño que tiene esta página web.

![Diseño Web](https://github.com/PortfolioJavierGil/Hate_BigData_Samsung-Innovation-Campus/assets/91088022/8f99f9cd-b2bb-4835-bd13-266e2dd075cb)


# Proyecto Realizado por:
* ## Alejandro Leo Ramírez
* ## Victor Tavara Pérez
* ## Javier Garrido Sola
* ## Javier Gil Rodríguez

