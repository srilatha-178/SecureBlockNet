import tensorflow as tf
from tensorflow import keras
from .attention import TemporalAttention

def build_secureblocknet(input_shape, num_classes=7, cnn_filters=(64,128), kernel_size=3,
                         bilstm_units=128, dense_units=(128,64), dropout=0.5,
                         learning_rate=1e-3, attention_units=64, return_attention_model=False):
    inp = keras.Input(shape=input_shape, name='traffic_sequence')
    x = inp
    for i, f in enumerate(cnn_filters):
        x = keras.layers.Conv1D(f, kernel_size, padding='same', activation='relu', name=f'conv1d_{i+1}')(x)
        x = keras.layers.BatchNormalization(name=f'bn_{i+1}')(x)
        if i == 0:
            x = keras.layers.MaxPooling1D(pool_size=2, padding='same', name='maxpool')(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(bilstm_units, return_sequences=True), name='bilstm'
    )(x)
    context, weights = TemporalAttention(attention_units=attention_units, return_attention=True, name='temporal_attention')(x)
    x = context
    for i, units in enumerate(dense_units):
        x = keras.layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
    probs = keras.layers.Dense(num_classes, activation='softmax', name='class_probabilities')(x)
    model = keras.Model(inp, probs, name='SecureBlockNet')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    if return_attention_model:
        attention_model = keras.Model(inp, [probs, weights], name='SecureBlockNetWithAttention')
        return model, attention_model
    return model

def training_callbacks(checkpoint_path, patience=10):
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience//3), min_lr=1e-6),
    ]
