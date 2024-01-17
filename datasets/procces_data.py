from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    clean_data = np.load("./datasets/EEG_all_epochs.npy")
    dirty_data = np.load("./datasets/EEG_contaminated_with_EOG_2db.npy")
    return clean_data, dirty_data


def clean_vs_dirty_EEG_graph(clean_data, dirty_data):
    # Escoge una muestra al azar
    index = np.random.randint(0, clean_data.shape[0])

    # Crea una figura y los ejes para el subplot
    fig, ax = plt.subplots()

    # Grafica la señal limpia y contaminada en la misma gráfica
    ax.plot(clean_data[index, :], label="EEG Limpio")
    ax.plot(dirty_data[index, :], label="EEG Contaminado")

    # Configura el título y las etiquetas de los ejes
    ax.set_title("Comparación de señales EEG")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Amplitud")

    # Muestra la leyenda
    ax.legend()

    plt.show()


def create_test_train_data(
    clean_data: np.ndarray,
    dirty_data: np.ndarray,
    tamanio_muestra_test: float = 0.2,
) -> tuple:
    split_idx = int(len(clean_data) * (1 - tamanio_muestra_test))

    # Divide los datos en conjuntos de entrenamiento y test
    clean_data_train = clean_data[:split_idx]
    dirty_data_train = dirty_data[:split_idx]

    clean_data_test = clean_data[split_idx:]
    dirty_data_test = dirty_data[split_idx:]

    return (
        clean_data_train,
        dirty_data_train,
        clean_data_test,
        dirty_data_test,
    )


def data_processing(
    clean_data_train, dirty_data_train, clean_data_test, dirty_data_test
):
    # Normalización de los datos
    scaler = StandardScaler()
    clean_data_train_normalizados = scaler.fit_transform(clean_data_train)
    dirty_data_train_normalizados = scaler.transform(dirty_data_train)
    clean_data_test_normalizados = scaler.transform(clean_data_test)
    datos_contamindos_de_prueba_normalizados = scaler.transform(dirty_data_test)

    # Reshape de los datos
    timesteps = 100
    n_samples, n_features = clean_data_train_normalizados.shape
    n_sequences = n_samples // timesteps
    clean_data_train_reshaped = np.reshape(
        clean_data_train_normalizados[: n_sequences * timesteps, :],
        (n_sequences, timesteps, n_features),
    )

    n_samples, n_features = dirty_data_train_normalizados.shape
    n_sequences = n_samples // timesteps
    dirty_data_train_reshaped = np.reshape(
        dirty_data_train_normalizados[: n_sequences * timesteps, :],
        (n_sequences, timesteps, n_features),
    )

    n_samples, n_features = clean_data_test_normalizados.shape
    n_sequences = n_samples // timesteps
    clean_data_test_reshaped = np.reshape(
        clean_data_test_normalizados[: n_sequences * timesteps, :],
        (n_sequences, timesteps, n_features),
    )

    n_samples, n_features = datos_contamindos_de_prueba_normalizados.shape
    n_sequences = n_samples // timesteps
    dirty_data_test_reshaped = np.reshape(
        datos_contamindos_de_prueba_normalizados[: n_sequences * timesteps, :],
        (n_sequences, timesteps, n_features),
    )

    return (
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    )


def prepare_data():
    # Se cargon los datos de la base de datos contaminada EEG y limpia
    clean_data, dirty_data = load_data()

    # Graficamos los datos en una comparación
    clean_vs_dirty_EEG_graph(clean_data, dirty_data)

    # Se crean los datos de prueba y entrenamiento
    (
        clean_data_train,
        dirty_data_train,
        clean_data_test,
        dirty_data_test,
    ) = create_test_train_data(clean_data, dirty_data)

    # Procesamos los datos para usarlos en las redes neuronales
    (
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    ) = data_processing(
        clean_data_train, dirty_data_train, clean_data_test, dirty_data_test
    )

    return (
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    )
