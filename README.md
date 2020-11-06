# Predicting winners: Academy Award for Best Actor

    This project sets out to settle the fierce debate surrounding Leonardo DiCaprio. Having received an oscard for his performance of Hugh Glass in the Hollywood rendition of "The Revenant" many critics and fans alike questioned the Academy's decision behind this award. Since then people have been wondering, namely, should he have won an oscar for one of his earlier performances? Here we seek to provide an answer to that question.
    
    To answer this question, I stumbled upon a secondary question that might help to inform the first: Does the bias implicit in critic reviews reflect the receiving of an award or nomination?
    
  
    
    
## The Data:
    
    The features used in this analysis are comprised of two ratings, the number of reviews made available by critics for each title, two scores done via sentiment analysis on the reviews available as a mean score for both the polarity(negative vs. positive) and subjectivity(objective vs. subjective), as well as an additional column where the actor's name has been preproccessed into a numeric value for the algorithm to weigh individual actor contribution on.
    
    The target's are represented as an award column. Here 0 corresponds to a movie in which the actor did not receive a nomination, 1 corresponds to the actor having received a nomination for Best Actor in that title, and 2 corresponds to an actor winning the award for Best Actor for that title.
    
    
    
    
## The Work:

    For the purposes of transparency I have included two notebooks comprising the work done. The first is Actor_data_cleaning and should be straightforward. Here the raw data was imported and cleaned using pandas, numpy, nltk, wordcloud, and textblob. The cleaned data set was then saved and used to conduct the analysis in the second notbook.
    The second notebook titled Leo_predictions shows the iterative process of using a grid search cross validation method for fine tuning the hyperparameters for the XGBoost tree based decision model on the whole of the data set. Having determined the hyperparameters of the model two different tests are conducted where an attempt to predict DiCaprio's nominations and wins.
    
    The first method uses a model trained on a random selection of data to make the predictions with.
    the second method uses a model trained on all but DiCaprio's data.
    
    
    
    
## Overview of Results:

    There is considerable variation in between predictions and the predictions and the observed values.
    
    The first model shows accurate predictions on several titles where DiCaprio received nominations: The Aviator, Blood Diamond and Once Upon a Time in Hollywood. It also predicted nominations for his performances in Spielberg, Inception, Catch Me if You Can, Titanic, and William Shakespeare's Romeo & Juliet.
    
    The second model has no accurate predictions for nominations/wins. It does predict that DiCaprio should have had nominations for Spielberg, Before the Flood, and William Shakespeare's Romeo & Juliet, as well as two wins for his performances in Inception and Catch Me if You Can.
    
    Furthermore, Iit is interesting to note that both models agree that he should have won an oscar for early career performance as Arnie Grape in What's Eating Gilbert Grape.
    
    
    
    
## Conclusion:
    This was a fun project where I attempted to answer a ridiculous question through scientific means.
    
    For future consideration I might look into how well the model performs for other actors e.g. Daniel Day-Lewis.
    
    
    
    