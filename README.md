# netflix-prize
Movie Recommendation System using Collaborative filtering on Netflix Prize Dataset (2006)
The algorithm uses a variant of SVD to perform collaborative filtering. Consider a large 
matrix where rows represent the users and columns represent movies. Each cell (i,j) of the 
matrix represents the rating given by ith user to the jth movie. Let us call this matrix R.
For Netflix Prize Dataset, the number of users is around 480k and movies around 17k. This
results in a matrix with over 8 billion cells. However, not all users rate all the movies.
Each user rates only a handful of movies. In the Netflix Prize Dataset, the number of ratings
available are around 100 million, which is around 1.25% of the total ratings in the matrix R.
Thus, the matrix R is sparse.

We try to factor the rating matrix, R, into two matrices, U and M, where U is of shape
(num_users, k) and M is of shape (num_movies, k). Here, k is the feature dimension. The
matrix U can be considered as a feature matrix for users, where each row represents the
choices of a single user. M can be viewed with the same analogy as U. The rating a user
would give to a movie, is directly proportional to similarity in feature dimension of
the user and the movie. The features can be thought of comprising of various genres such as
action, comedy, thriller, horror. If a user likes action and comedy, his feature vector will
represent this with a high value corresponding to these attributes and lower value for other
attributes. Similar is the case for movies. Thus, We can represent R as:
                $$R = UM^T$$

Here is a hyperparameter and different values of k  could be tried to reach the most optimal
results. In my implementation, I have set k=40.

Another factor to consider in predicting rating is the bias. Some users prefer to give an average
rating to movies they dislike or some may never give a very high rating to any movie. There might
be biases associated with movies as well. Movies with religious sentiments lie mostly on the extreme
ends of ratings. Thus, it makes sense to attach a bias with each user and movie.


## **How to run?**
1- Clone the repository and head inside the repository folder
2- Download the Netflix Prize Dataset. If you have kaggle api installed and setup, simply run
   ```sh run.sh data```
   If you do not have kaggle api set up, head to https://www.kaggle.com/netflix-inc/netflix-prize-data,
   and download and extract the data in the data folder. If you follow this step, manually create,
   output and logs directory in the main folder.
3- Train the data
   ```sh run.sh train [v1/v2/v3]```
   v1 - simple svd model without biases
   v2 - svd model with user and movie biases
   v3 - svd model with user, movie and global biases ( faster convergence than v2 )
4- Test the data
   ```sh run.sh test [v1/v2/v3]```
   The results are saved in output/v[1/2/3]_test_result.csv
5- Get recommendations for user
   ```sh run.sh recommend [v1/v2/v3] $user_id```
   where user_id is the id of user for which recommendation is required,
   The results are saved in output/v[1/2/3]_recommend_result.csv

