"""
## Summary

Entrada:

La entrada (input) es un tensor tridimensional con forma (100, 512, 1).

Encoder:

La primera capa convolucional (Conv2D) tiene 128 filtros, un tamaño de kernel de (3, 3), 
activación ReLU y relleno ('same' para mantener la forma de la entrada). Luego, 
se aplica una capa de MaxPooling con un tamaño de ventana (2, 2) para reducir las dimensiones espaciales a la mitad.
La segunda capa convolucional tiene 32 filtros con un tamaño de kernel de (3, 3), 
activación ReLU y el mismo relleno. Otra capa de MaxPooling reduce las dimensiones espaciales nuevamente a la mitad.

Decoder:

La primera capa deconvolucional (Conv2DTranspose) tiene 32 filtros con un tamaño de kernel de (3, 3), 
una tasa de paso de 2 y activación ReLU. Esto aumenta las dimensiones espaciales a la mitad.
La segunda capa deconvolucional tiene 64 filtros, un tamaño de kernel de (3, 3), 
una tasa de paso de 2 y activación ReLU. Esto aumenta aún más las dimensiones espaciales.
La última capa convolucional tiene 1 filtro, un tamaño de kernel de (3, 3), 
activación sigmoide y relleno ('same'). Esta capa produce la salida del modelo 
con un solo canal.

Compilación:

El modelo se compila utilizando el optimizador Adam y la función de pérdida de entropía 
cruzada binaria (binary_crossentropy). Este tipo de pérdida es comúnmente utilizado en 
problemas de reconstrucción de imágenes cuando se espera una salida binaria (por ejemplo, 
imágenes en escala de grises normalizadas entre 0 y 1).

Observaciones:

El modelo es un autoencoder, que se utiliza para comprimir y luego reconstruir datos de entrada. 
La compresión ocurre en la fase de encoder, y la reconstrucción en la fase de decoder.
La última capa utiliza una función de activación sigmoide, indicando que se espera una salida en 
el rango de 0 a 1, lo cual es apropiado para datos normalizados.
"""

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