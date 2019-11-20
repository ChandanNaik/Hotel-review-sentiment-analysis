# Hotel-review-sentiment-analysis
Classifying hotel review as either positive or negative using logistic regression 


Python Program focused on Natural Language Processing and Machine Learning (Binary Classification using Logistic Regression) to get the sentiment analysis of hotel reviews - either positive or negative.

About the included files:

The 'hotel_review_preprocessing.py' script is responsible for taking care of preprocessing the raw dataset(hotelPosT-train.txt, hotelNegT-train.txt, negative-words.txt, positive-words.txt and hotel-review-testset.txt).

The ' hotel_review_train_test.py' script imports preprocessed data, trains a text classification system for sentiment analysis using logistic regression, and pickles necessary Python objects for further use in the creation of the 'modelPrediction.txt'.


'modelPrediction.txt' will have the classification for each of the reviews in 'hotel-review-testset.txt'.
