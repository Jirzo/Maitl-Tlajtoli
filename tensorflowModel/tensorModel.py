import tensorflow as tf

# import numpy as np # Importa NumPy para operaciones numéricas, aunque no se usa directamente en este snippet, es común en proyectos de ML.
# from sklearn.model_selection import train_test_split # Importa una función de scikit-learn para dividir datos, aunque no se usa directamente aquí con TFRecord.

# --- Configuración de Dimensiones ---
# Estas variables definen la forma de los datos de entrada para el modelo.
timesteps = 30  # Número de "pasos de tiempo" o frames en cada secuencia de landmarks.
num_landmarks = 21  # Número de landmarks (puntos clave) detectados por mano (ej., MediaPipe Hands detecta 28).
features_per_landmark = (
    3  # Número de características por landmark (ej., x, y, z coordenadas).
)
input_dim = (
    num_landmarks * features_per_landmark
)  # Dimensión total de las características para un solo frame (28 * 3 = 63).


# --- Función para Parsear TFRecord ---
def _parse_function(example_proto):
    """
    Función de ayuda para parsear un solo 'tf.train.Example' de un TFRecord.
    Define cómo se leen las características 'landmarks' y 'label' del formato binario.

    Args:
        example_proto: Un string proto serializado que representa un tf.train.Example.

    Returns:
        Un tuple de tensores: (landmarks, label).
    """
    # Define la descripción de las características que se esperan en el TFRecord.
    # "landmarks": Se espera un array plano de floats con el tamaño total de la secuencia (timesteps * input_dim).
    # "label": Se espera un entero de 64 bits.
    feature_description = {
        "landmarks": tf.io.FixedLenFeature([timesteps * input_dim], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    # Parsea el ejemplo proto único utilizando la descripción de características.
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Remodela el tensor 'landmarks' a la forma deseada (timesteps, input_dim),
    # que es (30, 63) en este caso, representando 30 frames con 63 características cada uno.
    landmarks = tf.reshape(parsed_example["landmarks"], (timesteps, input_dim))
    # Extrae el tensor 'label'.
    label = parsed_example["label"]
    return landmarks, label


# --- Dataset Loader ---
def load_dataset(tfrecord_path, batch_size=32, shuffle=True):
    """
    Carga un dataset de TFRecord y lo prepara para el entrenamiento o la evaluación.

    Args:
        tfrecord_path (str): La ruta al archivo TFRecord.
        batch_size (int): El tamaño del batch de datos a devolver.
        shuffle (bool): Si True, el dataset se baraja.

    Returns:
        Un objeto tf.data.Dataset listo para ser usado por el modelo.
    """
    # Crea un dataset a partir del archivo TFRecord.
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    # Aplica la función de parseo a cada elemento del dataset.
    dataset = dataset.map(_parse_function)
    # Si se especifica, baraja el dataset. El buffer_size (1000) es el tamaño del buffer
    # desde el cual se seleccionan elementos aleatoriamente para el barajado.
    if shuffle:
        dataset = dataset.shuffle(1000)
    # Agrupa los elementos del dataset en batches del tamaño especificado.
    dataset = dataset.batch(batch_size)
    return dataset


def tf_trainer():
    # --- Cargar Datasets ---
    batch_size = 32  # Define el tamaño del batch a usar durante el entrenamiento y la evaluación.
    # Carga el dataset de entrenamiento desde el archivo TFRecord, con barajado activado.
    train_ds = load_dataset(
        "landmarks_escalados.tfrecord", batch_size=batch_size, shuffle=True
    )
    # Carga el dataset de prueba. Aquí se usa el mismo archivo TFRecord, pero sin barajar.
    # En un escenario real, deberías tener archivos TFRecord separados para entrenamiento y prueba.
    test_ds = load_dataset(
        "landmarks_escalados.tfrecord", batch_size=batch_size, shuffle=False
    )

    # --- Modelo LSTM ---
    # Define la arquitectura de la red neuronal, que es una red recurrente (LSTM).
    model = tf.keras.Sequential(
        [
            # La capa de entrada define la forma de los datos que el modelo espera recibir:
            # (timesteps, input_dim) -> (30, 63)
            tf.keras.layers.Input(shape=(timesteps, input_dim)),
            # Primera capa LSTM con 128 unidades. `return_sequences=True` significa que esta capa
            # devolverá la secuencia completa de outputs para la siguiente capa LSTM.
            tf.keras.layers.LSTM(128, return_sequences=True),
            # Segunda capa LSTM con 64 unidades. Esta capa no devuelve secuencias (`return_sequences` es False por defecto),
            # por lo que su output será la salida del último timestep, que se pasa a las capas Dense.
            tf.keras.layers.LSTM(64),
            # Una capa densa (totalmente conectada) con 64 neuronas y función de activación ReLU.
            tf.keras.layers.Dense(64, activation="relu"),
            # Capa de salida densa. El número de neuronas debe ser igual al número de clases.
            # La activación 'softmax' se usa para problemas de clasificación multiclase,
            # produciendo probabilidades para cada clase.
            # NOTA: Reemplaza "28" por el número real de clases de tu dataset.
            tf.keras.layers.Dense(28, activation="softmax"),
        ]
    )

    # Compila el modelo, configurando el optimizador, la función de pérdida y las métricas.
    model.compile(
        optimizer="adam",  # El optimizador Adam es una opción popular y eficiente.
        loss="sparse_categorical_crossentropy",  # Función de pérdida para clasificación multiclase cuando las etiquetas son enteros.
        metrics=[
            "accuracy"
        ],  # Métrica para monitorear durante el entrenamiento (la precisión).
    )

    # --- Callbacks ---
    # Los callbacks son funciones que se ejecutan en ciertas etapas del entrenamiento.
    # Callbacks para guardar el mejor modelo basado en la métrica de validación.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "mejor_modelo.keras",  # Nombre del archivo donde se guardará el modelo.
        monitor="val_accuracy",  # Métrica a monitorear para decidir cuál es el "mejor" modelo (precisión de validación).
        save_best_only=True,  # Solo guarda el modelo si la métrica monitoreada mejora.
        mode="max",  # La métrica 'val_accuracy' debe ser maximizada.
        verbose=1,  # Muestra mensajes de progreso.
    )

    # Callback para detener el entrenamiento tempranamente si la pérdida de validación deja de mejorar.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Métrica a monitorear (pérdida de validación).
        patience=5,  # Número de épocas sin mejora después de las cuales el entrenamiento se detendrá.
        restore_best_weights=True,  # Restaura los pesos del modelo a los de la mejor época.
        verbose=1,  # Muestra mensajes de progreso.
    )

    # --- Entrenamiento ---
    # Inicia el proceso de entrenamiento del modelo.
    # train_ds: Dataset de entrenamiento.
    # validation_data: Dataset de validación (se usa para evaluar el rendimiento del modelo en datos no vistos durante el entrenamiento).
    # epochs: Número máximo de veces que el modelo iterará sobre todo el dataset de entrenamiento.
    # callbacks: Lista de callbacks a usar durante el entrenamiento.
    history = model.fit(
        train_ds, validation_data=test_ds, epochs=30, callbacks=[checkpoint, early_stop]
    )

    # --- Evaluación después de Entrenar ---
    # Evalúa el rendimiento final del modelo en el dataset de prueba.
    test_loss, test_acc = model.evaluate(test_ds)
    # Imprime la precisión del modelo en el dataset de prueba.
    print(f"Accuracy en test: {test_acc:.2f}")
