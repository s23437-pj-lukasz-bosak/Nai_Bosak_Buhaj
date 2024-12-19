import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Autorzy: Mariusz Buhaj, Łukasz Bosak
# nalezy zainstalowac biblioteki: pip install tensorflow matplotlib seaborn scikit-learn
# train - trenuje nowy model i zapisuje na dysku.
# classify - uruchomienie scenariusza.

(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
"""
Ładowanie zbioru danych Fashion MNIST

Dane zawierają:
- train_images, train_labels: obrazy i etykiety do treningu,
- test_images, test_labels: obrazy i etykiety do testowania
"""

train_images = train_images / 255.0
test_images = test_images / 255.0
"""
Normalizacja wartości pikseli obrazów do zakresu [0, 1]
"""

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
"""
Zmiana kształtu danych obrazów, aby uwzględnić kanał (1 dla obrazów w skali szarości)
"""

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
"""
Lista nazw klas odpowiadających etykietom w zbiorze Fashion MNIST
"""

model_path = 'fashion_mnist_model.h5'
"""
Ścieżka do pliku, w którym zapisany jest model
"""

if os.path.exists(model_path):
    choice = input("Model istnieje na dysku. Chcesz go ponownie wytrenować, czy dokonać klasyfikacji? (train/classify): ").strip().lower()

    if choice == 'classify':
        model = load_model(model_path)
        """
        Wczytanie zapisanego modelu z pliku

        model : tensorflow.keras.models.Sequential
            Wczytany model sieci neuronowej
        """
        print("Model został wczytany z dysku")

        image_path = input("Podaj ścieżkę do obrazu do klasyfikacji: ").strip()
        try:
            image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
            """
            Ładowanie i przeskalowanie obrazu wejściowego

            image_path : str
                Ścieżka do obrazu
            image : PIL.Image.Image
                Załadowany obraz przeskalowany do wymiarów 28x28
            """
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array)
            """
            Predykcja klasy obrazu za pomocą modelu

            predictions : numpy.ndarray
                Wyniki predykcji dla każdej klasy
            """
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class]
            print(f"Przewidywana klasa: {predicted_class} ({predicted_class_name})")
        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania obrazu: {e}")
    else:
        retrain = True
else:
    retrain = True

if 'retrain' in locals() and retrain:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    """
    Definicja modelu sieci neuronowej dla klasyfikacji obrazów Fashion MNIST
    """

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    """
    Kompilacja modelu z optymalizatorem Adam i funkcją straty sparse_categorical_crossentropy
    """

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    """
    Trenowanie modelu na zbiorze treningowym

    history : tensorflow.python.keras.callbacks.History
        Historia treningu modelu
    """

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc:.2f}, test loss: {test_loss:.2f}')

    model.save(model_path)
    print(f"Model został zapisany do {model_path}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true = test_labels.flatten()

    cm = confusion_matrix(y_true, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
