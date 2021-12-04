import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding

class SVD(layers.Layer):
    def __init__(self, users, movies, latent_dim, **kwargs):
        super(SVD, self).__init__(**kwargs)
        self._users = users
        self._movies = movies
        self._latent_dim = latent_dim
        self._user_embedding = None
        self._movie_embedding = None

    def build(self, input_shape):
        self._user_embedding = Embedding(input_dim=self._users,
                                         output_dim=self._latent_dim,
                                         name='user_features'
                                        )
        self._movie_embedding = Embedding(input_dim=self._movies,
                                          output_dim=self._latent_dim,
                                          name='movie_features'
                                         )

    def call(self, u, m, training=False):
        u = self._user_embedding(u)
        m = self._movie_embedding(m)
        return tf.keras.layers.dot([u,m], axes=1, normalize=False)

class Bias(layers.Layer):
    def __init__(self, users, movies, trainable_bias=False, user_bias=None, movie_bias=None, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self._users = users
        self._movies = movies
        self._trainable_bias = trainable_bias
        self._user_bias = user_bias
        self._movie_bias = movie_bias

    def build(self, input_shape):
        if self._trainable_bias:
            self._user_bias = Embedding(input_dim=self._users,
                                        output_dim=1,
                                        name='user_bias'
                                       )
            self._movie_bias = Embedding(input_dim=self._movies,
                                         output_dim=1,
                                         name='movie_bias'
                                        )
        else:
            self._user_bias = Embedding(input_dim=self._users,
                                        output_dim=1,
                                        embeddings_initializer=tf.keras.initializers.Constant(self._user_bias),
                                        trainable=False,
                                        name='user_bias'
                                       )
            self._movie_bias = Embedding(input_dim=self._movies,
                                         output_dim=1,
                                         embeddings_initializer=tf.keras.initializers.Constant(self._movie_bias),
                                         trainable=False,
                                         name='movie_bias'
                                        )


    def call(self, u, m, training=False):
        u = self._user_bias(u)
        m = self._movie_bias(m)
        return tf.keras.layers.add([u,m])