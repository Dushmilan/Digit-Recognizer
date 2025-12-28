import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

import keras
from keras import layers, models, callbacks

# 1. Load the dataset
try:
    df = pd.read_csv('Data/mnist1.5k.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Data/mnist1.5k.csv' not found.")
    exit()

# 2. Data Preparation
# Note: Column names in your CSV are capitalized (e.g., 'Label')
label_col = 'Label' if 'Label' in df.columns else 'label'

y = df[label_col]
X = df.drop(label_col, axis=1)

# Reshape flattened pixel data back to 28x28 images
X = X.values.reshape(-1, 28, 28, 1)
y = y.values

# Normalization: Scale pixels to [0, 1] range
X = X.astype('float32') / 255.0

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 4. Define Data Augmentation as Layers (Modern Keras 3 way)
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
])

# 5. Build the Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    
    # First Conv Block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),

    # Dense Layers
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

# 6. Compile
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# 7. Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# 8. Train
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=30, 
    batch_size=32,
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 9. Evaluate
print("\nEvaluating on Test Set...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Visualization (Saving plots to files in case display is not available)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.savefig('training_curves.png')
print("Training curves saved as training_curves.png")

# Generate the confusion matrix array
cm = confusion_matrix(y_test, y_pred)

# Define labels for the plot (0-9)
class_names = [str(i) for i in range(10)]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.xlabel('Predicted Label (Model\'s Guess)')
plt.ylabel('True Label (Actual Digit)')
plt.title('Confusion Matrix: Where is the model failing?')
plt.savefig('confusion_matrix.png')
print("Detailed confusion matrix saved as confusion_matrix.png")

# 11. Save the Model
model.save('digit_recognizer_Model.h5')
print("\nSuccess: Model saved as digit_recognizer_Model.h5")
