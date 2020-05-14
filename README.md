# Disaster Response Web App

## 1. Project Motivation

This project was created to put my knowledge and skills in data engineering into practice. Specifically, an ETL pipeline and a NLP machine learning pipeline were created in python scripts to extract, transform, and store Twitter messages in response to disasters for the application of classification machine learning trainings. Finally, the machine learning model was deployed via a web application which takes text inputs from users and classifies the input message into one or multiple number of 35 pre-defined disaster categories.

This application can potentially help first responders and government agencies to understand the needs of the public and prioritize its rescue efforts accordingly after a disaster occurs by monitoring and making inferences from the social media.

## 2. File and Folder Structure

- [app](./app/)
  - `run.py`: python script for launching the web app
- [data](./data/)
  - `disaster_messages.csv`: Training data
  - `disaster_categories.csv`: Training data
  - `process_data.py`: ETL pipeline python script
  - `DisasterResponse.db`: Output of the ETL pipeline stored in SQLite database
- [models](./models/)
  - `train_classifier.py`: Machine learning pipeline python script
  - `classifier.pkl`: Machine learning model output
- [requirements.txt](./requirements.txt): Python packages used for this project

## 3. Installation/Setup

1. Run the following shell commands to set up the python environment and project workspace

   ```sh
   # clone the repo
   $ git clone https://github.com/caroger/disaster_response
   $ cd disaster_response/

   # create a virtual environment using Anaconda
   $ conda create -y --name disaster python=3.7
   $ pip install -r requirements.txt

   # install required packages
   $ conda activate disaster
   ```

2. Run the following shell commands in the project's root directory to set up your database and model.

   ```sh
   # To run ETL pipeline that cleans data and stores in database
   $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

   # To run ML pipeline that trains classifier and saves
   $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. Run the following command in the app's directory to run your web app

   ```sh
   cd app/
   python run.py
   ```

4. Launch the app on your browser <http://0.0.0.0:3001/>

## 4. Data Processing and Machine Learning Steps

### ETL Pipeline

1. split 36 response variables stored in 1 text string into 36 separated binary columns

2. removed the rows where the `related` response variable equal to value of 2 as these messages

3. dropped the column corresponding to the `child_alone` response variable due to 100% of its values are 0. In other words, it provides no merit in training the machine learning model

### ML Pipeline

1. Split the available training data into 80% training and 20% testing data

2. trimmed, stemmed, and tokenized the input text variable

3. performed text feature extractions with:

   - [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
   - [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

4. implemented regularized linear classification models with stochastic gradient descent learning ([SGD](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)).

5. used [randomized search cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to find the best estimator

### Results and Caveats

1. The [micro f1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) was used as the scoring benchmark for model selection and tuning due to the highly imbalanced nature of the training data as shown in the figure below.

![Response Variable Value Counts](/images/newplot.png)

2. The best performing model achieved a micro f1-score of 0.69

```
                        precision    recall  f1-score  support
related                  0.845729  0.932944  0.887198   3937.0
request                  0.717762  0.652655  0.683662    904.0
offer                    0.000000  0.000000  0.000000     29.0
aid_related              0.674767  0.773445  0.720744   2154.0
medical_help             0.542955  0.385366  0.450785    410.0
medical_products         0.563953  0.364662  0.442922    266.0
search_and_rescue        0.622642  0.207547  0.311321    159.0
security                 0.142857  0.009346  0.017544    107.0
military                 0.581395  0.441176  0.501672    170.0
water                    0.738602  0.738602  0.738602    329.0
food                     0.807763  0.774823  0.790950    564.0
shelter                  0.731771  0.621681  0.672249    452.0
clothing                 0.705882  0.461538  0.558140     78.0
money                    0.538462  0.304348  0.388889    115.0
missing_people           0.687500  0.152778  0.250000     72.0
refugees                 0.533333  0.341463  0.416357    164.0
death                    0.670051  0.501901  0.573913    263.0
other_aid                0.432432  0.303207  0.356470    686.0
infrastructure_related   0.386861  0.154070  0.220374    344.0
transport                0.635417  0.244000  0.352601    250.0
buildings                0.626263  0.457565  0.528785    271.0
electricity              0.561644  0.398058  0.465909    103.0
tools                    0.000000  0.000000  0.000000     32.0
hospitals                0.333333  0.066667  0.111111     60.0
shops                    0.000000  0.000000  0.000000     19.0
aid_centers              0.625000  0.078125  0.138889     64.0
other_infrastructure     0.322581  0.085106  0.134680    235.0
weather_related          0.785469  0.768612  0.776949   1491.0
floods                   0.827044  0.591011  0.689384    445.0
storm                    0.748344  0.666012  0.704782    509.0
fire                     0.476190  0.238095  0.317460     42.0
earthquake               0.879386  0.795635  0.835417    504.0
cold                     0.821429  0.377049  0.516854    122.0
other_weather            0.553191  0.177474  0.268734    293.0
direct_report            0.627081  0.564436  0.594111   1001.0
micro avg                0.733820  0.652608  0.690835  16644.0
```

## 5. Credit

This project was completed as part of and under the guidelines of [Udacity Data Science Nano Degree Program](https://www.udacity.com/school-of-data-science) and with the data provided by [Figure Eight](https://appen.com/)
