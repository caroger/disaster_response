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

## 3. Installation

1. Run the following shell commands to set up the python environment and project workspace

   ```sh
   # clone the repo
   $ git clone https://github.com/caroger/disaster_response
   $ cd disaster_response/

   # create a virtual environment using Anaconda
   $ conda create -y --name disaster python=3.7

   # install required packages
   $ conda install --force-reinstall -y -q --name disaster -c conda-forge --file requirements.txt
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
   cd app
   python run.py
   ```

4. Launch the app on your browser <http://0.0.0.0:3001/>

## 4. Data Processing and Machine Learning Steps

### ETL Pipeline

1. split 36 response variables stored in 1 text string into 36 separated binary columns

2. removed the rows having `related` response variable equal to value of 2 as these messages

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

1. The trained model has xxx testing in accuracy and xx in f1 score.

2. Micro-f1 score was used as the scoring benchmark for model selection and tuning due to the highly imbalanced nature of the training data.

## 5. Credit

This project was completed as part of and under the guidelines of [Udacity Data Science Nano Degree Program](https://www.udacity.com/school-of-data-science) and with the data provided by [Figure Eight](https://appen.com/)
