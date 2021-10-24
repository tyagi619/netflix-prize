import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dot

import Config

def readInputFile(filename):
    dataframe = []
    with open(filename, "r") as f:
        movieId = None
        while True:
            line = f.readline()
            if not line:
                break

            if line[-1] == '\n':
                line = line[:-1]
            
            if line[-1] == ':':
                movieId = int(line[:-1])
            else:
                userId = int(line.split(',')[0])
                rating = float(line.split(',')[1])
                dataframe.append([userId, movieId, rating])
    dataframe = np.array(dataframe)
    return pd.DataFrame(data=dataframe, columns=['User', 'Movie', 'Rating']).astype({'User':int, 
                                                                                     'Movie':int, 
                                                                                     'Rating':float}
                                                                                    )

def loadData():
    inputFiles = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
    df_list = []
    print('#'*100)
    for i,file in enumerate(inputFiles):
        print('Reading files: {}/{}'.format(i+1, len(inputFiles)))
        filepath = os.path.join(Config.get('DATADIR'), file)
        df = readInputFile(filepath)
        df_list.append(df)
    print('#'*100)

    df = pd.concat(df_list, ignore_index=True)
    
    print('Mapping Users....')
    unique_users = list(df['User'].unique())
    num_users = len(unique_users)
    map_users = {id.item():i for i,id in enumerate(unique_users)}
    df['User'] = df['User'].map(map_users)
    userMapFile = os.path.join(Config.get('OUTDIR'), Config.get('USER_MAP'))
    print('User Mapping Complete')
    print('Writing User Mapping to {}'.format(Config.get('USER_MAP')))
    with open(userMapFile, "w") as f:
        json.dump(map_users, f)
    print('#'*100)

    print('Mapping Movies....')
    unique_movies = list(df['Movie'].unique())
    num_movies = len(unique_movies)
    map_movies = {id.item():i for i,id in enumerate(unique_movies)}
    df['Movie'] = df['Movie'].map(map_movies)
    movieMapFile = os.path.join(Config.get('OUTDIR'), Config.get('MOVIE_MAP'))
    print('Movie Mapping Complete')
    print('Writing Movie Mapping to {}'.format(Config.get('MOVIE_MAP')))
    with open(movieMapFile, "w") as f:
        json.dump(map_movies, f)
    print('#'*100)

    X = df[['User', 'Movie']].values
    y = df['Rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    print('Saving validation set to file for testing...')
    np.savetxt(Config.get('X_TEST_FILEPATH'), X_test, fmt='%d', delimiter=',', newline='\n')
    np.savetxt(Config.get('Y_TEST_FILEPATH'), y_test, fmt='%.1f', delimiter=',', newline='\n')
    print('#'*100)

    x_train_user = tf.convert_to_tensor(X_train[:,0])
    x_train_movie = tf.convert_to_tensor(X_train[:,1])
    y_train = tf.convert_to_tensor(y_train)
    train_data = tf.data.Dataset.from_tensor_slices(({'user':x_train_user, 'movie':x_train_movie}, y_train))
    train_data = train_data.shuffle(4096, reshuffle_each_iteration=True)
    train_data = train_data.batch(2048)

    x_val_user = tf.convert_to_tensor(X_test[:,0])
    x_val_movie = tf.convert_to_tensor(X_test[:,1])
    y_val = tf.convert_to_tensor(y_test)
    val_data = tf.data.Dataset.from_tensor_slices(({'user':x_val_user, 'movie':x_val_movie}, y_val))
    val_data = val_data.batch(2048)
    return train_data, val_data, num_users, num_movies

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

    movie_bias = Embedding(input_dim=num_movies,
                           output_dim=1,
                           input_length=None,
                           name='movie_bias')(movie_input)
    user_bias = Embedding(input_dim=num_users,
                          output_dim=1,
                          input_length=None,
                          name='user_bias')(user_input)
    
    output = Dot(axes=1, normalize=False, name='rating')([x,y])
    output = tf.keras.layers.Add()([output, movie_bias, user_bias])
    return Model(inputs=[user_input, movie_input], outputs=output, name='recommender')

def train(train_data, val_data, num_users, num_movies, latent_dim=40):
    model = recommenderModel(num_users=num_users, 
                             num_movies=num_movies,
                             latent_dim=latent_dim)
    print(model.summary())
    print('#'*100)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )

    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=Config.get('CHECKPOINT'),
                                                         monitor='val_root_mean_squared_error',
                                                         save_best_only=True,
                                                         mode='min',
                                                         save_freq='epoch',
                                                         save_weights_only=True,
                                                         verbose=1
                                                        )
    history = model.fit(x=train_data, 
                        validation_data=val_data, 
                        epochs=10, 
                        callbacks=[modelCheckpoint],
                        verbose=1
                       )

def main():
    Config.set('CURDIR', os.getcwd())
    train_data, val_data, num_users, num_movies = loadData()
    train(train_data, val_data, num_users, num_movies)

if __name__ == '__main__':
    main()