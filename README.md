# NFL Game Predictor - In Progress


Initially, I looked at team stats for a given game. Specifically, I wanted to see what stats were correlated with each other and which stats are correlated with wins:

![univariate](/images/univariate.png)

![wins](/images/initial_variable_exploration/download.png)

![pred_wins](/images/initial_variable_exploration/download-1.png)



Next, to build a predictive model, I found the average of the home/away teams' previous 2,3,4,5, and 6 previous game stats and used the net of home and away teams as a predictive features of a win/loss. I made all predictions using a Logistical Regresssion Model. 

Here are different features prediciting a win for a 6 game time frame: 

![predictive](/images/basic_model_results/download-6.png)

![predictive](/images/confusion_pred_mat_all_vars.png)


## Coming Soon
<li> Use more granular data as predictors -> play by play data </li>
<li> Compare the performance of the Logistical Regression Model to a Random Forest Model </li>
<li> Train models using a test/train split and K-fold validation process </li>


