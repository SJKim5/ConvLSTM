import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_images(image_folder, start_year, end_year, start_month, end_month):
    images = []
    missing_files = 0
    for year in range(start_year, end_year + 1):
        for month in range(start_month if year == start_year else 1, end_month + 1 if year == end_year else 13):
            filename = f"chl_{year}_{month}.png"
            img_path = os.path.join(image_folder, filename)
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
            else:
                missing_files += 1
                logging.warning(f"Missing: {filename}")
    if missing_files > 0:
        logging.warning(f"Total missing files: {missing_files}")
    logging.info(f"Loaded {len(images)} images out of {217}.")
    return np.array(images)

# Function to create sequences from image arrays
def create_sequences(images, sequence_length):
    X, y = [], []
    for i in range(len(images) - sequence_length):
        X.append(images[i:i+sequence_length])
        y.append(images[i+sequence_length])
    logging.info("Sequences created.")
    return np.array(X), np.array(y)

# Build ConvLSTM model
def build_model(sequence_length, width, height, channels):
    model = Sequential([
        ConvLSTM2D(filters=128, kernel_size=(5, 5), activation='relu', input_shape=(sequence_length, width, height, channels), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=False),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(width * height * channels, activation='sigmoid'),
        Reshape((width, height, channels))
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    logging.info("Model built and compiled.")
    return model

# Parameters
image_folder = ""
train_start_year = 2006
train_end_year = 2023
sequence_length = 5
width, height, channels = 128, 128, 3 

# Load and prepare data
all_images = load_images(image_folder, train_start_year, train_end_year, 1, 12)
X, y = create_sequences(all_images, sequence_length)

# Time Series Split Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
    logging.info(f"Starting fold {fold_idx+1}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = build_model(sequence_length, width, height, channels)
    model.fit(X_train, y_train, epochs=10, batch_size=4)  # Adjust epochs and batch size as needed

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    mape = mean_absolute_error(y_test.flatten(), predictions.flatten()) * 100

    fold_results.append((mse, mape))
    logging.info(f"Fold {fold_idx+1} completed with MSE: {mse}, MAPE: {mape}")

average_mse = np.mean([res[0] for res in fold_results])
average_mape = np.mean([res[1] for res in fold_results])

logging.info(f"Cross-Validation Results: Average MSE: {average_mse}, Average MAPE: {average_mape}")

# Display the final test image and the corresponding predicted image
def display_images(real, predicted):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(real)
    ax[0].set_title('Test Image')
    ax[0].axis('off')  # Hide axes ticks

    ax[1].imshow(predicted)
    ax[1].set_title('Predicted Image')
    ax[1].axis('off')  # Hide axes ticks

    plt.show()
    
final_test_image = y_test[-1]
final_predicted_image = predictions[-1]
display_images(final_test_image, final_predicted_image)
