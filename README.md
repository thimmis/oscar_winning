# Predicting winners: Academy Award for Best Actor

This project sets out to settle the fierce debate surrounding Leonardo DiCaprio. Having received an oscar for his performance of Hugh Glass in the Hollywood rendition of "The Revenant" many critics and fans alike questioned the Academy's decision behind this award. Since then people have been wondering, namely, should he have won an oscar for one of his earlier performances? Here we seek to provide an answer to that question.

To help answer this quesion I asked a secondary question: "Is the bias implicit in critic reviews representative of the scores given to it:

I explore this idea a bit by looking into the language used by critics in a number of reviews available for each title.




## The Data:

The features used in this analysis are comprised of two ratings, the number of reviews made available by critics for each title, two scores done via sentiment analysis on the reviews available as a mean score for both the polarity(negative vs. positive) and subjectivity(objective vs. subjective), as well as an additional column where the actor's name has been preproccessed into a numeric value for the algorithm to weigh individual actor contribution on.

The target's are represented as an award column. Here 0 corresponds to a movie in which the actor did not receive a nomination, 1 corresponds to the actor having received a nomination for Best Actor in that title, and 2 corresponds to an actor winning the award for Best Actor for that title.




## The Work:

For the purposes of transparency I have included two notebooks and a dash/plotly app comprising the work done. The first is Actor_data_cleaning and should be straightforward. Here the raw data was imported and cleaned using pandas and numpy, then a basic sentiment analysis was performed using nltk, wordcloud, and textblob to produce mean values for each title. The cleaned data set was then saved and used to conduct the analysis in the second notbook.

The second notebook titled Leo_predictions showcases the work done to tune the hyperparameters for the model with 'roc_auc_ovo' as the scoring metric for the cross validation due to imbalanced representations in the target classes. A base model identifying this as a multiclass classification problem is used with a fixed learning rate, with the remaining parameters unspecified so as to use the default values. Using this base model a random search cv is performed to determine a good neighborhood for the values of: (max_depth, min_child_weight, colsample_bytree, and subsample). A model with these parameters is then fit to the data using XGBoost 'merror' to determine the optimal number of trees to produce, and is then used to predict the target classes and probabilities for the training and test sets. Then a fine tuning of the parameters is repeated via a grid search cv to determine the parameters within a given range.

This iterative process is continued progressively fine tuning the tree based parameters as well as the regularization parameters to control under/over fitting of the model with periodic recalibration of the optimal number of estimators--as t --> the model should improve and n_est should decrease.

Once the hyperparameters are decided two models are developed to predict DiCaprio's awards/nominations with. The first is a general model that uses a random train/test split to train the model on and verify the results with. Using this a third set containing a holdout of DiCaprio's filmography is then used to predict.

The second model uses exactly the same hyperparameters as in the general model but is trained on all but DiCaprio's filmography.

A final merge is done to compare the results of the two predictions.


A simple dashboard can be found here: https://best-actors.herokuapp.com/ 




## Overview of Results:

Using the micro weighted ROC AUC score provided a model that is fairly certain about the accuracy of the predictions it makes.

From:

Accuracy (Train): 94.34%
AUC Score (Train): 0.9872
Accuracy Score (Test): 48.65%
AUC Score (Train): 0.8262

on the base model, to:

Accuracy (Train): 70.12%
AUC Score (Train): 0.8828
Accuracy (Test): 56.11%
AUC Score (Train): 0.8659

while the accuracy and soundness of predictions decreases on the training set, considerable improvement is made to the accuracy and soundness of the predictions for the test set. 

In both cases the models over predict the number of nomination and awards based on the actual numbers for DiCaprio. Both models seem to agree on which movies DiCaprio should have/did receive a nomination for. The generalized model has predicts one more movie to have received an award or nomination than the specified model, and the specified model predicts that many of these should have been wins.

The overlap of these models is centered on films that are some of his more memorable or unique performances.

## Conclusion:

This was an attempt to answer a silly question that is highly subject to personal bias through scientific means. It was a fun project through which I have learned a great deal. 

These models seem to take into consideration the people's opinion slightly more than the opinions of critics. Furthermore the models assume that an actor's performance exists in a vacuum of sorts. It is likely that many of the over predicted nominations and awards are due to another actor's performance receiving a nomination for a different role in the same film.

However, it is interesting to note the overlap, to me this suggests that the models are not far off from being able to accurately predict. Additional information such as main vs supporting role might help to improve the number and accuracy of predictions.

## Further Consideration:

The data currently lacks important information regarding each actor's roles, most notably is the fact that they do not take into consideration leading/supporting roles. Which in the future this will be accounted for as a binary value.


    
