
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Set seed agar hasil eksperimen konsisten
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('1k_heart_attack_prediction_indonesia.csv')
df['alcohol_consumption'].fillna('None', inplace=True)

df_encoded = pd.get_dummies(df, columns=[
    'gender',
    'region',
    'income_level',
    'smoking_status',
    'alcohol_consumption',
    'physical_activity',
    'dietary_habits',
    'air_pollution_exposure',
    'stress_level',
    'EKG_results'
])

x = df_encoded.drop('heart_attack', axis=1)
y = df_encoded['heart_attack']

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

# Pertama, split data jadi train+val dan test
x_train_full, x_test, y_train_full, y_test = train_test_split(
    x_scaled, y, test_size=0.1, random_state=42
)

# Kedua, split lagi train_full jadi train dan val
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.3, random_state=42
)


model = Sequential()
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[early_stop]
)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'akurasi: {accuracy * 100:.2f}%')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
