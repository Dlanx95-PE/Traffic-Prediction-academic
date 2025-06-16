"""
Definition of NN models (LSTM, GRU, SAEs)
Compatible with modern TensorFlow/Keras versions (e.g., Keras 3+)
"""

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Activation, Input



def get_lstm(units):
    """
    Build LSTM model.

    Args:
        units (List[int]): [timesteps, lstm1_units, lstm2_units, output_units]

    Returns:
        model (Sequential): Compiled LSTM model
    """
    model = Sequential([
        LSTM(units[1], input_shape=(units[0], 1), return_sequences=True),
        LSTM(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='linear')  # 'linear' recommended for regression
    ])
    return model


def get_gru(units):
    """
    Build GRU model.

    Args:
        units (List[int]): [timesteps, gru1_units, gru2_units, output_units]

    Returns:
        model (Sequential): Compiled GRU model
    """
    model = Sequential([
        GRU(units[1], input_shape=(units[0], 1), return_sequences=True),
        GRU(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='linear')
    ])
    return model

"""
def _get_sae(inputs, hidden, output):

    Build a simple Autoencoder model.

    Args:
        inputs (int): Input dimension
        hidden (int): Hidden layer size
        output (int): Output layer size

    Returns:
        model (Sequential): Autoencoder model

    model = Sequential([
        Dense(hidden, input_dim=inputs, activation='sigmoid', name='hidden'),
        Dropout(0.2),
        Dense(output, activation='linear')
    ])
    return model
"""
def _get_sae(inputs, hidden, output):
    """
    Build a simple Autoencoder model using the Functional API.

    Args:
        inputs (int): Input dimension
        hidden (int): Hidden layer size
        output (int): Output layer size

    Returns:
        model (Model): Autoencoder model
    """
    inp = Input(shape=(inputs,))
    x = Dense(hidden, activation='sigmoid', name='hidden')(inp)
    x = Dropout(0.2)(x)
    out = Dense(output, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    return model

"""
def get_saes(layers):

    Build stacked autoencoders (SAEs).

    Args:
        layers (List[int]): [input_dim, h1, h2, h3, output_dim]

    Returns:
        models (List[Sequential]): [sae1, sae2, sae3, stacked_sae]

    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential([
        Dense(layers[1], input_dim=layers[0], activation='sigmoid', name='hidden1'),
        Dense(layers[2], activation='sigmoid', name='hidden2'),
        Dense(layers[3], activation='sigmoid', name='hidden3'),
        Dropout(0.2),
        Dense(layers[4], activation='linear')
    ])
"""
def get_saes(layers):
    """
    Build stacked autoencoders (SAEs).

    Args:
        layers (List[int]): [input_dim, h1, h2, h3, output_dim]

    Returns:
        models (List): [sae1, sae2, sae3, stacked_sae]
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    # Reemplazamos Sequential por un modelo funcional
    input_layer = Input(shape=(layers[0],))
    hidden1 = Dense(layers[1], activation='sigmoid', name='hidden1')(input_layer)
    hidden2 = Dense(layers[2], activation='sigmoid', name='hidden2')(hidden1)
    hidden3 = Dense(layers[3], activation='sigmoid', name='hidden3')(hidden2)
    dropout = Dropout(0.2)(hidden3)
    output_layer = Dense(layers[4], activation='linear')(dropout)

    saes = Model(inputs=input_layer, outputs=output_layer)

    return [sae1, sae2, sae3, saes]