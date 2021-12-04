#!/bin/bash

if [ "$1" = "train" ]; then
    if [ "$2" = "v1" ]; then
        if ! [ -d output/v1 ]; then
            mkdir output/v1
        fi
        if ! [ -d output/v1/model ]; then
            mkdir output/v1/model
        fi
        python3 run.py train v1 --save-to=./output/v1/model --probe-src=./data/probe.txt --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt --save-xval-to=./output/v1/x_test.csv --save-yval-to=./output/v1/y_test.csv --save-user-map-to=./output/v1/user.json --save-movie-map-to=./output/v1/movie.json
    elif [ "$2" = "v2" ]; then
        if ! [ -d output/v2 ]; then
            mkdir output/v2
        fi
        python3 run.py train v2 --latent-dim=40 --max-epoch=10 --save-to=./output/v2/model/best --probe-src=./data/probe.txt --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt --save-xval-to=./output/v2/x_test.csv --save-yval-to=./output/v2/y_test.csv --save-user-map-to=./output/v2/user.json --save-movie-map-to=./output/v2/movie.json
    elif [ "$2" = "v3" ]; then
        if ! [ -d output/v3 ]; then
            mkdir output/v3
        fi
        python3 run.py train v3 --latent-dim=40 --max-epoch=10 --save-to=./output/v3/model/best --probe-src=./data/probe.txt --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt --save-xval-to=./output/v3/x_test.csv --save-yval-to=./output/v3/y_test.csv --save-user-map-to=./output/v3/user.json --save-movie-map-to=./output/v3/movie.json        
    elif [ "$2" = "v4" ]; then
        if ! [ -d output/v4 ]; then
            mkdir output/v4
        fi
        python3 run.py train v4 --latent-dim=40 --max-epoch=10 --use-sigmoid=1 --save-to=./output/v4/model/best --probe-src=./data/probe.txt --train-src=./data/combined_data_1.txt,./data/combined_data_2.txt,./data/combined_data_3.txt,./data/combined_data_4.txt --save-xval-to=./output/v4/x_test.csv --save-yval-to=./output/v4/y_test.csv --save-user-map-to=./output/v4/user.json --save-movie-map-to=./output/v4/movie.json
    else
        echo "Invalid model version selected"
    fi
elif [ "$1" = "test" ]; then
    if [ "$2" = "v1" ]; then
        python3 run.py test v1 ./output/v1/model ./output/v1//user.json ./output/v1/movie.json ./output/v1/x_test.csv ./output/v1/y_test.csv ./output/v1/v1_test_result.csv
    elif [ "$2" = "v2" ]; then
        python3 run.py test v2 --latent-dim=40 ./output/v2/model/best ./output/v2/user.json ./output/v2/movie.json ./output/v2/x_test.csv ./output/v2/y_test.csv ./output/v2/v2_test_result.csv
    elif [ "$2" = "v3" ]; then
        python3 run.py test v3 --latent-dim=40 ./output/v3/model/best ./output/v3/user.json ./output/v3/movie.json ./output/v3/x_test.csv ./output/v3/y_test.csv ./output/v3/v3_test_result.csv         
    elif [ "$2" = "v4" ]; then
        python3 run.py test v4 --latent-dim=40 --use-sigmoid=1 ./output/v4/model/best ./output/v4/user.json ./output/v4/movie.json ./output/v4/x_test.csv ./output/v4/y_test.csv ./output/v4/v4_test_result.csv
    else
        echo "Invalid model version selected"
    fi
elif [ "$1" = "recommend" ]; then
    if [ "$2" = "v1" ]; then
        python3 run.py recommend v1 ./output/v1/model ./output/v1/user.json ./output/v1/movie.json $3 ./output/v1/v1_recommend_result.csv
    elif [ "$2" = "v2" ]; then
        python3 run.py recommend v2 --latent-dim=40 ./output/v2/model/best ./output/v2/user.json ./output/v2/movie.json $3 ./output/v2/v2_recommend_result.csv
    elif [ "$2" = "v3" ]; then
        python3 run.py recommend v3 --latent-dim=40 ./output/v3/model/best ./output/v3/user.json ./output/v3/movie.json $3 ./output/v3/v3_recommend_result.csv         
    elif [ "$2" = "v4" ]; then
        python3 run.py recommend v4 --latent-dim=40 --use-sigmoid=1 ./output/v4/model/best ./output/v4/user.json ./output/v4/movie.json $3 ./output/v4/v4_recommend_result.csv
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