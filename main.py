from sklearn.preprocessing import StandardScaler
from neural_networks.cnn import Model as CNNModel
from neural_networks.lstm import Model as LSTMModel
from datasets.procces_data import prepare_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def graficar_resultados_modelo(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def generar_prediccion(Model, dirty_data_test_reshaped, clean_data_test_reshaped):
    # Crear listas para almacenar los resultados de cada métrica
    mse_values = []
    rmse_values = []
    mae_values = []
    corr_values = []

    scaler = StandardScaler()

    # Primero, genera algunas predicciones a partir de tus datos de prueba
    y_pred = Model.predict(dirty_data_test_reshaped)
    y_pred = np.squeeze(y_pred)
    print(y_pred.shape)

    # Bucle sobre cada señal en los datos de prueba
    for signal_number in range(dirty_data_test_reshaped.shape[0]):
        # Obtén la señal limpia real y la señal limpia predicha para esta señal específica
        real_clean_signal = clean_data_test_reshaped[signal_number]
        predicted_clean_signal = y_pred[signal_number]

        # Invierte la normalización de los datos
        real_clean_signal_inv = scaler.inverse_transform(real_clean_signal)
        predicted_clean_signal_inv = scaler.inverse_transform(predicted_clean_signal)

        # Selecciona un solo registro EEG
        real_clean_signal_single = real_clean_signal_inv[0, :]  # Primer registro
        predicted_clean_signal_single = predicted_clean_signal_inv[
            0, :
        ]  # Primer registro

        # Calcular MSE
        mse = mean_squared_error(
            real_clean_signal_single, predicted_clean_signal_single
        )
        mse_values.append(mse)

        # Calcular RMSE
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

        # Calcular MAE
        mae = mean_absolute_error(
            real_clean_signal_single, predicted_clean_signal_single
        )
        mae_values.append(mae)

        # Calcular Correlación de Pearson
        corr, _ = pearsonr(real_clean_signal_single, predicted_clean_signal_single)
        corr_values.append(corr)

    # Calcular la media de cada métrica
    mse_mean = np.mean(mse_values)
    rmse_mean = np.mean(rmse_values)
    mae_mean = np.mean(mae_values)
    corr_mean = np.mean(corr_values)

    print("Mean MSE:", mse_mean)
    print("Mean RMSE:", rmse_mean)
    print("Mean MAE:", mae_mean)
    print("Mean Pearson's correlation:", corr_mean)

    # Grafica los datos
    plt.figure(figsize=(12, 6))
    plt.plot(real_clean_signal_single, label="Real clean signal")
    plt.plot(predicted_clean_signal_single, label="Predicted clean signal")
    plt.legend()
    plt.title("Comparison of real and predicted clean signals")
    plt.show()


def run_cnn_model(
    clean_data_train_reshaped,
    dirty_data_train_reshaped,
    clean_data_test_reshaped,
    dirty_data_test_reshaped,
):
    history = CNNModel.fit(
        x=dirty_data_train_reshaped,
        y=clean_data_train_reshaped,
        epochs=50,
        batch_size=128,
        shuffle=False,
        validation_data=(
            dirty_data_test_reshaped,
            clean_data_test_reshaped,
        ),
    )

    graficar_resultados_modelo(history)


def run_lstm_model(
    clean_data_train_reshaped,
    dirty_data_train_reshaped,
    clean_data_test_reshaped,
    dirty_data_test_reshaped,
):
    history = LSTMModel.fit(
        dirty_data_train_reshaped,
        clean_data_train_reshaped,
        epochs=50,
        batch_size=128,
        validation_data=(dirty_data_test_reshaped, clean_data_test_reshaped),
    )

    graficar_resultados_modelo(history)


def run():
    # Se cargan y procesan los datos de entrenamiento
    (
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    ) = prepare_data()

    # Correr el modelo CNN
    run_cnn_model(
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    )

    # Correr el modelo LSTM
    run_lstm_model(
        clean_data_train_reshaped,
        dirty_data_train_reshaped,
        clean_data_test_reshaped,
        dirty_data_test_reshaped,
    )


run()
