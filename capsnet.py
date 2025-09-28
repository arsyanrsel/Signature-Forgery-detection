import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras import layers, models, backend as K

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

class PrimaryCaps(layers.Layer):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=dim_capsule * n_channels,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = tf.reshape(output, (-1, self.n_channels * output.shape[1] * output.shape[2], self.dim_capsule))
        return squash(output)

class DigitCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                        self.input_dim_capsule, self.dim_capsule],
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='W')

    def call(self, inputs, **kwargs):
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 4)
        W_expand = tf.expand_dims(self.W, 0)
        u_hat = tf.reduce_sum(inputs_expand * W_expand, axis=-2)

        b = tf.zeros_like(u_hat[..., 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c[..., None] * u_hat, axis=1)
            v = squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v[:, None, :, :], axis=-1)

        return v

def build_capsnet(input_shape=(224, 224, 1), routings=3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=9, strides=1, activation='relu')(inputs)
    x = PrimaryCaps(8, 32, kernel_size=9, strides=2, padding='valid')(x)
    x = DigitCaps(num_capsule=1, dim_capsule=16, routings=routings)(x)  
    x = Length()(x)
    outputs = layers.Activation('sigmoid')(x) 
    return models.Model(inputs, outputs)