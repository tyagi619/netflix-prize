# netflix-prize
Movie Recommendation System using Collaborative filtering on Netflix Prize Dataset (2006)<br>
The algorithm uses a variant of SVD to perform collaborative filtering. Consider a large<br> 
matrix where rows represent the users and columns represent movies. Each cell (i,j) of the<br> 
matrix represents the rating given by ith user to the jth movie. Let us call this matrix R.<br>
For Netflix Prize Dataset, the number of users is around 480k and movies around 17k. This<br>
results in a matrix with over 8 billion cells. However, not all users rate all the movies.<br>
Each user rates only a handful of movies. In the Netflix Prize Dataset, the number of ratings<br>
available are around 100 million, which is around 1.25% of the total ratings in the matrix R.<br>
Thus, the matrix R is sparse.<br>

We try to factor the rating matrix, R, into two matrices, U and M, where U is of shape<br>
(num_users, k) and M is of shape (num_movies, k). Here, k is the feature dimension. The<br>
matrix U can be considered as a feature matrix for users, where each row represents the<br>
choices of a single user. M can be viewed with the same analogy as U. The rating a user<br>
would give to a movie, is directly proportional to similarity in feature dimension of<br>
the user and the movie. The features can be thought of comprising of various genres such as<br>
action, comedy, thriller, horror. If a user likes action and comedy, his feature vector will<br>
represent this with a high value corresponding to these attributes and lower value for other<br>
attributes. Similar is the case for movies. Thus, We can represent R as:<br>
                $R = UM^T$

Here is a hyperparameter and different values of k  could be tried to reach the most optimal<br>
results. In my implementation, I have set k=40.<br>

Another factor to consider in predicting rating is the bias. Some users prefer to give an average<br>
rating to movies they dislike or some may never give a very high rating to any movie. There might<br>
be biases associated with movies as well. Movies with religious sentiments lie mostly on the extreme<br>
ends of ratings. Thus, it makes sense to attach a bias with each user and movie.<br>


## **How to run?**<br>
1- Clone the repository and head inside the repository folder<br>
2- Download the Netflix Prize Dataset. If you have kaggle api installed and setup, simply run<br>
   ```sh run.sh data```<br>
   If you do not have kaggle api set up, head to https://www.kaggle.com/netflix-inc/netflix-prize-data,<br>
   and download and extract the data in the data folder. If you follow this step, manually create,<br>
   output and logs directory in the main folder.<br>
3- Train the data<br>
   ```sh run.sh train [v1/v2/v3]```<br>
   v1 - simple svd model without biases<br>
   v2 - svd model with user and movie biases<br>
   v3 - svd model with user, movie and global biases ( faster convergence than v2 )<br>
4- Test the data<br>
   ```sh run.sh test [v1/v2/v3]```<br>
   The results are saved in output/v[1/2/3]_test_result.csv<br>
5- Get recommendations for user<br>
   ```sh run.sh recommend [v1/v2/v3] $user_id```<br>
   where user_id is the id of user for which recommendation is required,<br>
   The results are saved in output/v[1/2/3]_recommend_result.csv<br>
