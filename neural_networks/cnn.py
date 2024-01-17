import tensorflow as tf

input = tf.keras.layers.Input(shape=(100, 512, 1))
# Encoder
datos_limpios = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(input)
datos_limpios = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(datos_limpios)
datos_limpios = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(datos_limpios)
datos_limpios = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(datos_limpios)

# Decoder
datos_limpios = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(datos_limpios)
datos_limpios = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(datos_limpios)
datos_limpios = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(datos_limpios)

# Autoencoder
Model = tf.keras.Model(input, datos_limpios)
Model.compile(optimizer="adam", loss="binary_crossentropy")