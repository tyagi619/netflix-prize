"""
Usage:
    run.py train --train-src=<file> [options]
    run.py recommend [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py test [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --train-src=<file>                      train source file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 2048]
    --latent-dim=<int>                      latent dimension [default: 40]
    --max-epoch=<int>                       max epoch [default: 10]
    --patience=<int>                        wait for how many iterations to end training [default: 2]
    --lr=<float>                            learning rate [default: 0.001]
    --use-bias=<int>                        include user and movie bias paramters in learning [default: 0]
    --use-global-bias=<int>                 include global bias parameter in learning [default: 0]
    --save-to=<file>                        model save path [default: ./output/model]
    --save-xval-to=<file>                   src validation data save path [default: ./output/x_test.csv]
    --save-yval-to=<file>                   tgt validation data save path [default: ./output/y_test.csv]
    --save-user-map-to=<file>               user mapping save path [default: ./output/user.json]
    --save-movie-map-to=<file>              movie mapping save path [default: ./output/movie.json]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import logging
from docopt import docopt

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Model import Recommender

def readFile(filename):
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
                                                                                     'Rating':float
                                                                                    })

def loadData(inputFiles):
    df_list = []
    logging.info('reading input file...')
    for i,file in enumerate(inputFiles):
        logging.info('reading file: %d/%d', i+1, len(inputFiles))
        df = readFile(file)
        df_list.append(df)
    logging.info('reading input files complete')
    df = pd.concat(df_list, ignore_index=True)
    logging.info('dataset contains %d rows', df.shape[0])
    print('='*100)
    return df

def preprocessData(dataframe, args):
    logging.info('mapping users to continuous series...')
    unique_users = dataframe['User'].unique().tolist()
    num_users = len(unique_users)
    user_mapping = {id:i for i, id in enumerate(unique_users)}
    dataframe['User'] = dataframe['User'].map(user_mapping)
    logging.info('mapping users complete')

    logging.info('saving user mapping to disk...')
    userMappingFile = args['--save-user-map-to']
    with open(userMappingFile, "w") as f:
        json.dump(user_mapping, f)
    logging.info('saved user mapping to %s', args['--save-user-map-to'])
    print('='*100)

    logging.info('mapping movies to continuous series...')
    unique_movies = dataframe['Movie'].unique().tolist()
    num_movies = len(unique_movies)
    movie_mapping = {id:i for i, id in enumerate(unique_movies)}
    dataframe['Movie'] = dataframe['Movie'].map(movie_mapping)
    logging.info('mapping movies complete')

    logging.info('saving movie mapping to disk...')
    movieMappingFile = args['--save-movie-map-to']
    with open(movieMappingFile, "w") as f:
        json.dump(movie_mapping, f)
    logging.info('saved movie mapping to %s', args['--save-movie-map-to'])
    print('='*100)

    return dataframe, num_users, num_movies

def getTrainTestData(dataframe, batch_size, args):
    logging.info('splitting data into train-test...')
    X = dataframe[['User', 'Movie']].values
    y = dataframe['Rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    logging.info('train-test split complete')
    logging.info('X_train contains %d rows', X_train.shape[0])
    logging.info('X_val contains %d rows', X_test.shape[0])

    logging.info('writing validation data to disk...')
    np.savetxt(args['--save-xval-to'], X_test, fmt='%d', delimiter=',', newline='\n')
    np.savetxt(args['--save-yval-to'], y_test, fmt='%.1f', delimiter=',', newline='\n')
    logging.info('write validation data to %s and %s complete', args['--save-xval-to'], args['--save-yval-to'])
    print('='*100)

    logging.info('converting train data to tensorflow Dataset....')
    x_train_user = tf.convert_to_tensor(X_train[:,0])
    x_train_movie = tf.convert_to_tensor(X_train[:,1])
    y_train = tf.convert_to_tensor(y_train)
    train_data = tf.data.Dataset.from_tensor_slices(({'users':x_train_user, 'movies':x_train_movie}, y_train))
    train_data = train_data.shuffle(2*batch_size, reshuffle_each_iteration=True)
    train_data = train_data.batch(batch_size)
    logging.info('train data converted to tensorflow Dataset')

    logging.info('converting validation data to tensorflow Dataset...')
    x_val_user = tf.convert_to_tensor(X_test[:,0])
    x_val_movie = tf.convert_to_tensor(X_test[:,1])
    y_val = tf.convert_to_tensor(y_test)
    val_data = tf.data.Dataset.from_tensor_slices(({'users':x_val_user, 'movies':x_val_movie}, y_val))
    val_data = val_data.batch(batch_size)
    logging.info('validation data converted to tensorflow Dataset')
    print('='*100)
    
    return train_data, val_data

def train(args):

    if args['--train-src']:
        train_files = args['--train-src'].split(',')
    else:
        logging.error('no input files provided for training')
        raise RuntimeError('No input files for training.Use --train-src argument to pass input files')
    
    if len(train_files)==0:
        logging.error('empty input file list passed')
        raise RuntimeError('No input files for training.Use --train-src argument to pass input files')

    train_batch_size = int(args['--batch-size'])
    logging.info('batch size: %d', train_batch_size)
    latent_dim = int(args['--latent-dim'])
    logging.info('latent dim: %d', latent_dim)
    max_epochs = int(args['--max-epoch'])
    logging.info('max epochs: %d', max_epochs)
    patience = int(args['--patience'])
    logging.info('patience: %d', patience)
    lr = float(args['--lr'])
    logging.info('learning rate: %.4f', lr)
    use_bias = bool(int(args['--use-bias']))
    logging.info('use bias: %d', use_bias)
    use_global_bias = bool(int(args['--use-global-bias']))
    logging.info('use global bias: %d', use_global_bias)
    model_save_path = args['--save-to']
    print('='*100)

    dataframe = loadData(train_files)
    dataframe, num_users, num_movies = preprocessData(dataframe, args)
    train_data, val_data = getTrainTestData(dataframe, train_batch_size, args)

    model = Recommender(users=num_users, 
                        movies=num_movies,
                        latent_dim=latent_dim,
                        use_bias=use_bias,
                        global_bias=use_global_bias)
    print(model.summary())
    print('='*100)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )

    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                         monitor='val_root_mean_squared_error',
                                                         save_best_only=True,
                                                         mode='min',
                                                         save_freq='epoch',
                                                         save_weights_only=True,
                                                         verbose=1
                                                        )
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                     mode='min',
                                                     min_delta=0.01,
                                                     patience=patience,
                                                     verbose=1)
    
    history = model.fit(x=train_data, 
                        validation_data=val_data, 
                        epochs=max_epochs, 
                        callbacks=[modelCheckpoint, earlyStopping],
                        verbose=1
                       )

def recommend(args):
    pass

def test(args):
    pass

def main():

    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler('logs/logs.log',mode="w"),
                            logging.StreamHandler(sys.stdout)
                        ]
                       )

    seed = int(args['--seed'])
    tf.random.set_seed(seed)

    if args['train']:
        train(args)
    elif args['recommend']:
        recommend(args)
    elif args['test']:
        test(args)
    else:
        logging.error('invalid run mode. expected [train/test/recommend]')
        raise RuntimeError('invalid run mode')
    
if __name__ == '__main__':
    main()