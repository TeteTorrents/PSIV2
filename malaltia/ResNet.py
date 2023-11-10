import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras import layers, models
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Dimensions de les imatges
img_height = 64
img_width = 64
img_channels = 3

# Estableix una llavor per a la reproducibilitat
np.random.seed(42)

# Defineix el Bloc Residual
def bloc_residual(x, filtres, tamany_kernel=3, stride=1, activacio='relu'):
    y = layers.Conv2D(filtres, tamany_kernel, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activacio)(y)
    y = layers.Conv2D(filtres, tamany_kernel, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Si les mides d'entrada i sortida no coincideixen, afegix una convolució 1x1
    if x.shape[-1] != filtres or stride != 1:
        x = layers.Conv2D(filtres, 1, strides=stride, padding='same')(x)
    
    # Afegeix l'entrada i la sortida
    y = layers.Add()([x, y])
    y = layers.Activation(activacio)(y)
    return y

# Defineix el model ResNet
def construeix_resnet(forma_entrada, num_classes):
    inputs = layers.Input(shape=forma_entrada)

    # Capa de convolució inicial
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Blocs residuals
    num_blocs_lista = [2, 2, 2]  # Nombre de blocs residuals a cada grup
    filtres_lista = [64, 128, 256]  # Nombre de filtres a cada grup

    for i, num_blocs in enumerate(num_blocs_lista):
        for j in range(num_blocs):
            stride = 2 if j == 0 and i != 0 else 1
            x = bloc_residual(x, filtres_lista[i], stride=stride)

    # Global Average Pooling i capa Dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='resnet')
    return model

# Ruta a les carpetes d'imatges i al fitxer CSV amb etiquetes
data_folder = r"C:\Users\marti\Downloads\CNN\AnnotatedPatches"
csv_path = r"C:\Users\marti\Downloads\CNN\AnnotatedPatches\AnnotatedPatches\window_metadata.csv"

# Llegeix el CSV per obtenir els noms d'imatges i etiquetes
df = pd.read_csv(csv_path)
image_names = df['ID'].values
labels = df['Presence'].values

# Inicialitza arrays per emmagatzemar dades d'imatges i etiquetes
X = []
y = []

# Càrrega d'imatges i etiquetes
for image_name, label in zip(image_names, labels):
    # Construeix la ruta completa de la imatge
    image_path = os.path.join(data_folder, f"{image_name}.png")  # Suposant imatges en format PNG

    # Carrega la imatge i redimensiona si és necessari
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)

    X.append(img_array)
    y.append(label)

# Converteix les llistes a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Converteix les etiquetes a codificació one-hot
y = to_categorical(y, num_classes=2)

# Divisió de les dades en conjunts d'entrenament i proves
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimeix les formes per verificar
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Definició de la validació creuada K-Fold
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Itera sobre els pliegues
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'Plegat {fold + 1}/{k_folds}')

    # Divisió de les dades
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Construcció del model ResNet
    model = construeix_resnet(forma_entrada=(img_height, img_width, img_channels), num_classes=2)

    # Compilació del model 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['recall', 'precision', 'accuracy'])

    # Entrenament del model 
    model.fit(X_fold_train, y_fold_train, epochs=10, batch_size=32, validation_data=(X_fold_val, y_fold_val))

    # Avaluació del model en el conjunt de proves
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Precisió de proves pel plegat {fold + 1}: {test_acc}')

    # model.save(f'model_fold_{fold + 1}.h5')
