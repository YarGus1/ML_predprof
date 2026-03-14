import tensorflow as tf #для обучения
import tensorflow_datasets as tfds 
import numpy as np # Для работы с массивами
import matplotlib.pyplot as plt # Для отображения изображений
from tensorflow.keras.preprocessing import image # Для работы с изображениями
import os 
import shutil # для работы с директориями

# --- 1. Настройки ---
IMG_SIZE = 96  # Уменьшено для ускорения на CPU
BATCH_SIZE = 12  # Уменьшено для экономии памяти
EPOCHS = 10  # Уменьшено для быстрого теста
DATA_NAME = 'cats_vs_dogs' # Название датасета
CACHE_DIR = '/home/yargus/tensorflow_datasets' # Путь до кэша с датасетом

# --- 2. Очистка кэша (если поврежден) ---
dataset_path = os.path.join(CACHE_DIR, DATA_NAME) # Создание пути до датасета обучения
if os.path.exists(dataset_path):
    if not os.path.exists(os.path.join(dataset_path, 'dataset_info.json')): # Проверяет есть ли файл dataset_info.json
        print("⚠️ Обнаружен поврежденный кэш. Удаляем...")
        shutil.rmtree(dataset_path) # Удаляет файл если он поврежден

# --- 3. Загрузка данных ---
print("Загрузка датасета... (первый раз может занять 5-10 минут)") 
try:
    dataset, info = tfds.load(
        DATA_NAME, 
        with_info=True, 
        as_supervised=True,
        split=['train[:80%]', 'train[80%:]'], #
        data_dir=CACHE_DIR 
    )
    train_ds, val_ds = dataset  
    print(f"✅ Датасет загружен. Классы: {info.features['label'].names}") # вывод 
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    print("Попробуйте: pip install --upgrade tensorflow-datasets")
    exit(1)

# --- 4. Предобработка ---
def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0 #преобразование в вещественное числ
    return image, label

train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 5. Создание модели ---
print("Создание модели...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, 
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 6. Обучение ---
print("Обучение модели...")
history = model.fit(
    train_ds, 
    epochs=EPOCHS, 
    validation_data=val_ds
)

# --- 7. Сохранение ---
model.save('cats_vs_dogs_model.h5')
print("✅ Модель сохранена в 'cats_vs_dogs_model.h5'")

# --- 8. Тестирование ---
def predict_my_image(model, path):
    if not os.path.exists(path):
        print(f"⚠️ Файл {path} не найден")
        return
    
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Собака" if prediction > 0.5 else "🐱 Кошка"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\n📸 Результат: {label}")
    print(f"📊 Уверенность: {confidence:.2%}")

if __name__ == "__main__":
    test_file = 'dog.jpg'
    if os.path.exists(test_file):
        predict_my_image(model, test_file)
    else:
        print(f"\n💡 Для проверки поместите файл '{test_file}' в папку со скриптом")