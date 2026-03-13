import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 100

# Загружаем готовую модель
model = tf.keras.models.load_model('cats_vs_dogs_model.h5')

def predict_my_image(model, path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Собака" if prediction > 0.5 else "🐱 Кошка"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"Результат: {label}")
    print(f"Уверенность: {confidence:.2%}")

# Использование
predict_my_image(model, 'cat.jpg')