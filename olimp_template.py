import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Подготовка данных (NumPy)
# X_train, y_train должны быть numpy массивами


# Нормализация: 
X_train = X_train / 255.0 

# 2. Построение модели (Keras)
def create_model(input_shape, num_classes):
    # Пример: Простая CNN или MLP
    inputs = keras.Input(shape=input_shape)
    
    # База (например, для картинок)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Регуляризация
    x = layers.Dropout(0.5)(x)
    
    # Выход
    if num_classes == 1: # Бинарная
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss_fn = 'binary_crossentropy'
    else: # Многоклассовая
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss_fn = 'sparse_categorical_crossentropy'
        
    model = keras.Model(inputs, outputs)
    return model, loss_fn

model, loss_fn = create_model(input_shape=(32, 32, 3), num_classes=10)

# 3. Компиляция
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn,
    metrics=['accuracy']
)

# 4. Callbacks (Спасение от переобучения)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# 5. Обучение
history = model.fit(
    X_train, y_train,
    validation_split=0.2, # Если нет отдельного валидационного набора
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. Предсказание
predictions = model.predict(X_test)