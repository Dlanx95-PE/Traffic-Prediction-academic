"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)
    """
    print(f"📌 Entrenando modelo: {name.upper()}...")

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        verbose=1  # Esto muestra el progreso por época
    )

    # Guardar modelo entrenado
    model.save(f'model/{name}.h5')
    model.save(f'model/{name}.keras')

    # Guardar historial de pérdida en CSV
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(f'model/{name}_loss.csv', encoding='utf-8', index=False)

    # Graficar pérdida y guardar imagen
    plt.figure()
    plt.plot(hist.history['loss'], label='Loss (entrenamiento)')
    plt.plot(hist.history['val_loss'], label='Loss (validación)')
    plt.title(f'Pérdida del modelo {name.upper()}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'model/{name}_loss.png')
    plt.close()

    print(f"✅ Modelo '{name}' entrenado y guardado.")
    print(f"📉 Última pérdida de entrenamiento: {hist.history['loss'][-1]:.4f}")
    print(f"📉 Última pérdida de validación: {hist.history['val_loss'][-1]:.4f}")
    print('-' * 40)

def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(inputs=p.input,
                                       outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": 200}
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
