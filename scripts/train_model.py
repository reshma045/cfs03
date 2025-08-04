from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load Data
X, y = load_data("data/VR")

# Preprocess
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Define Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(y.shape[1])
])
model.compile(optimizer=Adam(1e-3), loss='mse')

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/bc_model.h5")

# Plot
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.savefig("results/training_plot.png")
plt.show()
