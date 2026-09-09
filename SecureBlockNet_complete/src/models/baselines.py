from tensorflow import keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def svm_baseline():
    return SVC(C=10.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)

def rf_baseline():
    return RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=-1, random_state=42)

def cnn_baseline(input_shape, num_classes=7):
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Conv1D(64,3,padding='same',activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2,padding='same'),
        keras.layers.Conv1D(128,3,padding='same',activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64,activation='relu'), keras.layers.Dropout(.5),
        keras.layers.Dense(num_classes,activation='softmax')])
    m.compile(keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy',['accuracy']); return m

def lstm_baseline(input_shape, num_classes=7):
    m=keras.Sequential([keras.Input(shape=input_shape),keras.layers.LSTM(128),keras.layers.Dropout(.5),keras.layers.Dense(num_classes,activation='softmax')])
    m.compile(keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy',['accuracy']); return m

def cnn_lstm_baseline(input_shape, num_classes=7):
    m=keras.Sequential([keras.Input(shape=input_shape),keras.layers.Conv1D(64,3,padding='same',activation='relu'),keras.layers.MaxPooling1D(2,padding='same'),keras.layers.LSTM(128),keras.layers.Dropout(.5),keras.layers.Dense(num_classes,activation='softmax')])
    m.compile(keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy',['accuracy']); return m

def bilstm_baseline(input_shape, num_classes=7):
    m=keras.Sequential([keras.Input(shape=input_shape),keras.layers.Bidirectional(keras.layers.LSTM(128)),keras.layers.Dropout(.5),keras.layers.Dense(num_classes,activation='softmax')])
    m.compile(keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy',['accuracy']); return m
