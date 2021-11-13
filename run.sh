#!/bin/bash

if [ "$1" = "train" ]; then
    if [ "$2" = "v1" ]; then
        python3 run.py train --latent-dim=2 --max-epoch=2 --save-to=./output/v1/model --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt
    elif [ "$2" = "v2" ]; then
        python3 run.py train --latent-dim=2 --max-epoch=2 --save-to=./output/v2/model --use-bias=1 --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt        
    elif [ "$2" = "v3" ]; then
        python3 run.py train --latent-dim=2 --max-epoch=2 --save-to=./output/v3/model --use-bias=1 --use-global-bias=1 --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt
    else
        echo "Invalid model version selected"
    fi
elif [ "$1" = "test" ]; then
    if [ "$2" = "v1" ]; then
        python3 run.py test --latent-dim=40 ./output/v1/model ./output/user.json ./output/movie.json ./output/x_test.csv ./output/y_test.csv ./output/v1_test_result.csv
    elif [ "$2" = "v2" ]; then
        python3 run.py test --latent-dim=40 --use-bias=1 ./output/v2/model ./output/user.json ./output/movie.json ./output/x_test.csv ./output/y_test.csv ./output/v2_test_result.csv         
    elif [ "$2" = "v3" ]; then
        python3 run.py test --latent-dim=40 --use-bias=1 --use-global-bias=1 ./output/v3/model ./output/user.json ./output/movie.json ./output/x_test.csv ./output/y_test.csv ./output/v3_test_result.csv
    else
        echo "Invalid model version selected"
    fi
elif [ "$1" = "recommend" ]; then
    if [ "$2" = "v1" ]; then
        python3 run.py recommend --latent-dim=40 --map-input=1 ./output/v1/model ./output/user.json ./output/movie.json $3 ./output/v1_recommend_result.csv
    elif [ "$2" = "v2" ]; then
        python3 run.py recommend --latent-dim=40 --map-input=1 --use-bias=1 ./output/v2/model ./output/user.json ./output/movie.json $3 ./output/v2_recommend_result.csv         
    elif [ "$2" = "v3" ]; then
        python3 run.py recommend --latent-dim=10 --map-input=1 --use-bias=1 --use-global-bias=1 ./output/v3/model ./output/user.json ./output/movie.json $3 ./output/v3_recommend_result.csv
    else
        echo "Invalid model version selected"
    fi
elif [ "$1" = "data" ]; then
    if [ -d data ]; then
        rm -r data
    fi
    mkdir data
    kaggle datasets download -d netflix-inc/netflix-prize-data -p data
    unzip data/netflix-prize-data.zip -d data
    rm data/netflix-prize-data.zip

    if ! [ -d output ]; then
        mkdir output
    fi
    if ! [ -d logs ]; then
        mkdir logs
    fi  
else
    echo "Invalid mode selected"
fi