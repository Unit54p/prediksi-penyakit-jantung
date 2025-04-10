# ========================================
# 1. Import Library yang Dibutuhkan
# ========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ========================================
# 2. Load Dataset dari CSV
# ========================================
df = pd.read_csv('heart_attack_prediction_indonesia.csv')

# ========================================
# 3. Tangani Missing Values
# ========================================
df['alcohol_consumption'].fillna('None', inplace=True)

# print(df.select_dtypes(include='object').nunique())  # cek jumlah nilai unik kolom bertipe objek (komentar dari kamu)

# ========================================
# 4. One-Hot Encoding: Mengubah Kolom Kategori Jadi Angka
# ========================================
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

# ========================================
# 5. Pisahkan Fitur (X) dan Label (Y)
# ========================================
# Pisahkan fitur (X) dan label (y)
x = df_encoded.drop('heart_attack', axis=1)
y = df_encoded['heart_attack']

# ========================================
# 6. Normalisasi Fitur
# ========================================
# Normalisasi nilai-nilai X dengan MinMaxScaler agar berada di rentang 0-1
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# ubah hasil normalisasi ke dalam bentuk DataFrame agar mudah dibaca
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.1, random_state=42
)


# inisialisasi model
model = Sequential()
# model.add(Dense(16, activation='relu', input_shape = (x_train.shape[1],)))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model = Sequential()
# model.add(Dense(16, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(Dropout(0.3))  # 30% neuron akan "mati" saat training
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# compile model
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

history = model.fit(
    x_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop]
)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'akurasi: {accuracy * 100:.2f}%')


# History dari model.fit()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
