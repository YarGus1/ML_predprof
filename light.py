import numpy as np
import matplotlib.pyplot as plt # Красивые картинки
from PIL import Image # Подготовка ото
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 # Датасет
from tensorflow.keras.models import Sequential # Класс для создания слоёв
from tensorflow.keras.layers import Flatten, Dense # Классы для создания связанных слоёв
from tensorflow.keras.utils import to_categorical # Вектоор в матрицу двиочных классов

def create_model():
    (x_train,y_train),(x_val,y_val) = cifar10.load_data()
    x_train = x_train / 255 # Подготовка фото

    x_val = x_val/255 # Подготовка валид. фото

    y_train = to_categorical(y_train,10) # Разметка тестовых классов данных на 10 категорий
    y_val = to_categorical(y_val,10) # Разметка валидационных классов на 10 категорий

    model = Sequential([
        Flatten(input_shape=(32,32,3)), # 1 слой для изображения 32*32 пикселя, 3 канала RGB
        Dense(1000, activation = 'relu'), # Слой 1000 нейронов, активация по алгоритму релю
        Dense(10, activation = 'softmax') # Последний слой 10 т.к. 10 классов
    ])

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=64, epochs = 10, validation_data = (x_val,y_val))
    model.save('cifar10_model.h5')

def main():


    image = Image.open('cat.jpg')
    resized = image.resize((32, 32))
    img_array = np.array(resized) / 255  # нормализуем значение пикселей
    img_array = img_array.reshape((1, 32, 32, 3))  # изображений, размеры, каналов

    model = tf.keras.models.load_model('cifar_model.h5')
    predictions = model.predict(img_array)

    classes = [
        'самолёт', 'автомобиль', 'птица', 'кошка', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик'
    ]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, predictions[0], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('Вероятность')
    ax.set_title('Что это')

    plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # встраивание диаграммы в веб-приложение

    plt.close()
if __name__ == "__main__":
    #create_model()
    main()