# Netflix-Prize
Movie Recommendation System using Collaborative filtering on Netflix Prize Dataset (2006)

The following model is based on the blog by [Simon Funk](https://sifter.org/~simon/journal/20061211.html)

## About the Dataset
Netflix Prize was a competition started by Netflix in 2006 to improve upon their existing recommender system, Cinematch. The goal of the competition was to improve upon the RMSE performance of Cinematch by 10% (RMSE ~0.855). Netflix released the dataset for the competition which consisted of actual ratings by different users. The dataset contined about 100M ratings by about 500K users and for 17.7K movies. The dataset guaranteed that each user has rated atleast one movie and that each movie has been rated by atleast one user. In addition to the train dataset, Netflix released a probe dataset which consisted of around 1.4M ratings. The probe dataset is a subset of train dataset. Other than the train and probe dataset, Netflix released the quiz data, the output for which was never released. The quiz data was used to calculate the final leaderboard position. Netflix also released a separate movie metadata file, that consisted of movie names and year of release.

## Baseline model (v1)
A simple intuitive approach for recommendation system would be to use the average ratings<br> of movies and users to predict the rating of unrated movies. Let <i>b<sub>i</sub></i> represent the average rating of <i>ith</i> movie and <i>b<sub>j</sub></i> represent the rating offset from average movie rating for <i>jth</i> user. One would think to simply average over the ratings available for a movie to calculate the average rating for movie. But consider a movie that has been rated only by a single user. The average of such a movie is highly biased. Thus, we use [<i>Bayesian Average</i>](https://en.wikipedia.org/wiki/Bayesian_average) to calculate average. If <i>R<sub>a</sub></i> and <i>V<sub>a</sub></i> are the mean and variance of all movies' average ratings (which defines prior expectation for a new movie's average rating before you have observed any actual rating) and <i>V<sub>b</sub></i> is the average variance of individual movie ratings (which tells how indicative each new observation is of the true mean), then,<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;K=V_{b}/V_{a}" title="Equation" /><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;mean=\frac{R_{a}\times%20K+sum(observedRatings)}{K+count{observedRatings}}" title="Equation" /><br><br>
<i>K</i> is a hyperparameter and can be experimented with different values. In my model, I use <i>K=25</i>. Using the above Bayesian average, we can compute a better estimate of the mean of individual movie rating and individual user rating offset. Thus, we can write <i>b<sub>i</sub></i> :<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;globalRatingAvg=\frac{sum(allRatings)}{count(allRatings)}" title="Equation" /><br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;b_{i}=\frac{globalRatingAvg\times%20K+sum(observedRatings)}{K+count(observedRatings)}" title="Equation" /><br><br>

And similarily for <i>b<sub>j</sub></i> :<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;offset_{ji}=r_{ji}-b_{i}" title="Equation" /><br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;globalOffsetAvg=\frac{sum(allOffset)}{count(allOffset)}" title="Equation" /><br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;b_{j}=\frac{globalOffsetAvg\times%20K+sum(observedOffset)}{K+count(observedOffset)}" title="Equation" /><br><br>

## SVD (v2)
The baseline method though quite simple has obvious drawbacks, the major one being that every user is recommended same set of movies. Thus, it makes sense to add some complexity to the model, that would adapt the system to suggest movies based on individual preferences and not only on average movie ratings.<br>
Consider the ratings as a large matrix, <i>R</i>, of shape (<i>num_users</i>, <i>num_movies</i>). We can factorize this matrix into 2 matrices, <i>U</i> and <i>M</i>, using SVD:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;R=UM^{T}" title="Equation" /><br>
where, shape of <i>U</i> is (<i>num_users</i>, <i>num_features</i>) and <i>M</i> is (<i>num_movies</i>, <i>num_features</i>). These <i>num_features</i> represent the amount of distinct features in movie matrix, and preference of users for each feature in user matrix.

Note that simple SVD breaks a matrix into 3 sub-matrices:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;R=U\Sigma%20V^{T}" title="Equation" /><br>
But <img src="https://latex.codecogs.com/svg.latex?\small&space;\Sigma" title="Equation" /> can be fused into the adjacent matrices.

The SVD looks great as a approach, but there is only one issue, we don't have the complete matrix to perform SVD. There are about 8.5B total cells in the matrix out of which we only know values for ~100M cells (quite sparse). To tackle this problem, we can use Gradient Descent to compute the matrices <i>U</i> and <i>M</i>, using MSE (square of RMSE) as the loss function.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;\hat{r}_{ji}=u_{j}m_{i}^{T}" title="Equation" /><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;err=(r_{ji}-\hat{r}_{ji})^{2}" title="Equation" /><br>

Other than adapting better to individual user preference, SVD has another advantage in terms of model size, where instead of 8.5B parameters, it only uses 20M paramters.


## SVD + Baseline (v3)
Combining the Bayesian and SVD approach, one can improve on the RMSE. The equations for this looks like:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;\hat{r}_{ji}=u_{j}v_{i}^{T}+b_{i}+b_{j}" title="Equation" /><br>

One could try to experiment with fine-tuning <i>b<sub>i</sub></i> and <i>b<sub>j</sub></i> while training, or keep them fixed. In the experiments I ran, keeping <i>b<sub>i</sub></i> and <i>b<sub>j</sub></i> fixed performed better on probe set, because it prevented overfitting for users and movies with small number of ratings available.
The results for this experiments match with those mentioned by [Simon Funk](https://sifter.org/~simon/journal/20061211.html) in his blog (Probe RMSE: ~0.932). Simon Funk suggested adding L2 regularization to feature vectors. I did not use regularization but instead used early stopping to prevent overfitting. However, L2 regularization could give a few decimal of improvement over the probe dataset.

## Adding Non-linerity (v4)
SVD is in some sense a linear model and linear models could be limiting. One way to deal with this could be to add a non-linear function to the final prediction. I used sigmoid as a non-linear function, but you could try experimenting with different non-linear functions.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.latex?\small&space;\hat{r}_{ji}=5.0\times%20\sigma(u_{j}v_{i}^{T}+b_{i}+b_{j})" title="Equation" /><br>

## Additional Exploration
As suggested earlier, one could experiment with using L2 regularization on the feature vectors to improve RMSE on probe data. Another addition to the existing model could be to incorporate movie release date into the model. This can be done by adding a temporal paramter, <img src="https://latex.codecogs.com/svg.latex?\small&space;\mu%20(t)" title="Equation" />. The idea is that ratings of movies change with time, movies in certain era are more liked by people, taste of people change with time. One fun thing to try would be to pass the feature vectors for user and movie into a neural network. The neural network here would serve as the non-linear function.

## **How to run?**<br>
1- Clone the repository and head inside the repository folder<br>
2- Download the Netflix Prize Dataset. If you have kaggle api installed and setup, simply run<br>
   ```sh run.sh data```<br>
   If you do not have kaggle api set up, head to https://www.kaggle.com/netflix-inc/netflix-prize-data,<br>
   and download and extract the data in the data folder. If you follow this step, manually create,<br>
   output and logs directory in the main folder.<br>
3- Train the data<br>
   ```sh run.sh train [v1/v2/v3/v4]```<br>
4- Test the data<br>
   ```sh run.sh test [v1/v2/v3/v4]```<br>
   The results are saved in output/[v1/v2/v3/v4]/test_result.csv<br>
5- Get recommendations for user<br>
   ```sh run.sh recommend [v1/v2/v3/v4] $user_id```<br>
   where user_id is the id of user for which recommendation is required,<br>
   The results are saved in output/[v1/v2/v3/v4]/recommend_result.csv<br>
