import random
import copy
import pickle
import shutil
import os
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg

#Funciones activadoras
def relu(z):
    
    a = np.maximum(0,z)

    cache = z

    return a, cache

def sigmoid(z):

    a = 1 / (1 + np.exp(-z))

    cache = z
    
    return a, cache

def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
 
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

# Function to load and preprocess images
def load_images_from_folder(folder, label, image_size, max_images=350):
    images = []
    labels = []
    filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    # Shuffle filenames to randomize the selection
    random.shuffle(filenames)
    
    # Select a subset of filenames
    selected_filenames = filenames[:max_images]
    
    for filename in selected_filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            img_array = np.array(img) / 255.0  # Normalize the images
            if img_array.shape == (image_size[0], image_size[1], 3):  # Check if the shape is correct
                images.append(img_array)
                labels.append(label)
            else:
                print(f"Skipping {filename}, incorrect shape: {img_array.shape}")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    return images, labels



def preprocess_data():

    image_size = (96,96)
    # Load images
    training_cat = 'datasets/working/train/Cat'
    training_not_cat = 'datasets/working/train/NotCat'

    cat_images, cat_labels = load_images_from_folder(training_cat, 1, image_size)
    not_cat_images, not_cat_labels = load_images_from_folder(training_not_cat, 0, image_size)

    # Combine and split the data manually
    all_images = cat_images + not_cat_images
    all_labels = cat_labels + not_cat_labels

    # Convert to numpy arrays
    all_set_x_orig = np.array(all_images, dtype=np.float32) # your train set features
    all_set_y_orig = np.array(all_labels, dtype=np.float32) # your train set labels

    # Split the data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(all_set_x_orig, all_set_y_orig, test_size=0.2, random_state=42)

    print(train_x.shape)
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y

"""
# Configuraciones
base_dir = 'dataset/'
train_dir = os.path.join("datasets/working/train")
val_dir = os.path.join("datasets/working/val")
test_dir = os.path.join("datasets/working/test")

#creating directory for each category

for category in ['Cat', 'NotCat']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

#función para dividir los datos
def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_SIZE=0.8):
    files = []
    for filename in os.listdir(SOURCE):
        file = os.path.join(SOURCE, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(f"{filename} is zero length, so ignoring.")   

    training_length = int(len(files) * SPLIT_SIZE)
    validation_length = int(len(files) * ((1 - SPLIT_SIZE) / 2))
    testing_length = int(len(files) - training_length - validation_length)


    #Bajaramos los ficheros
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    validation_set = shuffled_set[training_length:training_length + validation_length]
    testing_set = shuffled_set[training_length + validation_length:]

    for filename in training_set:
        shutil.copy(os.path.join(SOURCE, filename), os.path.join(TRAINING, filename))

    for filename in validation_set:
        shutil.copy(os.path.join(SOURCE, filename), os.path.join(VALIDATION, filename))

    for filename in testing_set:
        shutil.copy(os.path.join(SOURCE, filename), os.path.join(TESTING, filename))


split_data(os.path.join(base_dir, 'Cat'), os.path.join(train_dir, 'Cat'), os.path.join(val_dir, 'Cat'), os.path.join(test_dir, 'Cat'))
split_data(os.path.join(base_dir, 'NotCat'), os.path.join(train_dir, 'NotCat'), os.path.join(val_dir, 'NotCat'), os.path.join(test_dir, 'NotCat'))
"""
import matplotlib.pyplot as plt


def debug_labels(images, labels, num_samples=6):
    """
    Muestra una selección de imágenes junto con sus etiquetas asociadas.

    :param images: Lista o array de imágenes.
    :param labels: Lista o array de etiquetas correspondientes a las imágenes.
    :param num_samples: Número de imágenes a mostrar.
    """
    num_samples = min(num_samples, len(images))
    labels_t = labels.transpose()

    indices = np.random.choice(len(images), num_samples, replace=False)

    plt.figure(figsize=(15, 5))  # Ajusta el tamaño de la figura si es necesario
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Label: {labels_t[idx]}")
        plt.axis('off')
    plt.show()


def vector_to_image(vector, image_size, num_channels, normalized=True):
    """
    Convierte un vector a una imagen con las dimensiones especificadas.

    :param vector: Vector de características a convertir.
    :param image_size: Tamaño de la imagen (alto, ancho).
    :param num_channels: Número de canales de la imagen.
    :param normalized: Si las imágenes están normalizadas en el rango [0, 1].
    :return: Imagen reconvertida en formato (alto, ancho, canales).
    """
    image = vector.reshape((image_size[0], image_size[1], num_channels))
    
    if normalized:
        # Escala de vuelta al rango [0, 1] para visualizar correctamente si están normalizadas
        image = image * 255
    
    # Verifica el rango de los píxeles
    print("Pixel values range:", np.min(image), np.max(image))
    print(image.shape)
    print("Data type before conversion:", image.dtype)
    
    
    return image.astype(np.uint8)

def predict(X, y, parameters):
    """
    Hace predicciones usando el modelo y calcula la precisión.

    :param X: Datos de entrada con forma (n_features, m).
    :param y: Etiquetas verdaderas con forma (1, m).
    :param parameters: Parámetros del modelo.
    :return: Predicciones del modelo.
    """
    m = X.shape[1]  # Número de ejemplos
    print(f"Number of examples: {m}")
    n = len(parameters) // 2  # Número de capas en la red neuronal
    
    # Inicializa las predicciones
    p = np.zeros((1, m))
    
    # Propagación hacia adelante
    probas, caches = L_model_forward(X, parameters)
    
    # Convertir probabilidades a predicciones binarias
    p = (probas > 0.5).astype(int)
    
    # Cálculo de precisión
    accuracy = np.sum(p == y) / m

    # Imprimir resultados
    #print("Predicciones:")
    #print(p)
    #print("Etiquetas verdaderas:")
    #print(y)
    print(f"Precisión: {accuracy:.2f}")
    """ 
    image_size = (96,96)
    num_channels = 3
     # Mostrar imágenes cuando la predicción falla
    for i in range(m):
        if p[0, i] != y[0, i]:
            print(f"Prediction failed at index {i}. Showing the image with the correct label:")
            print(X[:,i] * 255)
            #print(vector_to_image(X[:,i], image_size, num_channels))
            plt.imshow(vector_to_image(X[:, i], image_size, num_channels))
            plt.title(f"True label: {y[0, i]}, Predicted: {p[0, i]}")
            plt.axis('off')
            plt.show()
    """
    return p


def preprocess_image(image_path, image_size=(96, 96)):
    """
    Load and preprocess an image from the given path.
    Args:
    image_path: str - Path to the image file
    image_size: tuple - Size to which the image will be resized

    Returns:
    np.array - Preprocessed image ready for prediction
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(image_size)
        
        # Mostrar la imagen
        plt.imshow(img)
        plt.axis('off')  # Opcional: para ocultar los ejes
        plt.show()
        
        img_array = np.array(img) / 255.0  # Normalize the image
        
        if img_array.shape == (image_size[0], image_size[1], 3):
            img_flatten = img_array.reshape(-1, 1)
            print(img_flatten.shape)
            return img_flatten
        else:
            raise ValueError(f"Incorrect image shape: {img_array.shape}")
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")