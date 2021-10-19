import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dot

import Config

def recommenderModel(num_users, num_movies, latent_dim=40):
    user_input = Input(shape=(), name='user')
    movie_input = Input(shape=(), name='movie')
    x = Embedding(input_dim=num_users,
                  output_dim=latent_dim,
                  input_length=None,
                  name='user_matrix')(user_input)
    y = Embedding(input_dim=num_movies,
                  output_dim=latent_dim,
                  input_length=None,
                  name='movie_matrix')(movie_input)
    output = Dot(axes=1, normalize=False, name='rating')([x,y])
    return Model(inputs=[user_input, movie_input], outputs=output, name='recommender')

def loadTestData():
    X_test = np.loadtxt(Config.get('X_TEST_FILEPATH'), delimiter=',')
    y_test = np.loadtxt(Config.get('Y_TEST_FILEPATH'))
    
    X_test = {'user':X_test[:,0], 'movie':X_test[:,1]}
    return X_test, y_test    

def main():
    Config.set('CURDIR', os.getcwd())
    X_test, y_test = loadTestData()
    model = recommenderModel(num_users=480189, num_movies=17770)
    model.load_weights(filepath=Config.get('CHECKPOINT')).expect_partial()
    y_pred = model.predict(x=X_test, batch_size=2048)
    
    result = pd.DataFrame({'y_pred':y_pred[:,0], 'y_test':y_test})
    result['diff'] = abs(result['y_pred'] - result['y_test'])
    x = result[result['diff'] < 0.5]
    print(result.head(50))
    print(x.shape)  # 4.7M ratings are within 0.5 error out of 10M

if __name__ == '__main__':
    main()