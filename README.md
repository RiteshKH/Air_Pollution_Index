# Air_Pollution_Index
Model to determine air pollution index based on factors like humidity, temperature, traffic volume etc. HackerEarth Challenge

## Feature sets in training data
* date, time with an hour interval
* holiday
* humidity
* wind_speed
* wind_direction
* visibility_in_miles
* dew_point
* temperature
* rain_p_h
* snow_p_h
* clouds_all
* weather_type
* traffic_volume

## Feature engineering
1. Checked and removed outliers from the numerical columns's data. Box-plot showed extreme data points in temperature, rain_p_h. Removed them and checked again. Rain and snow showed outliers, which can be managed by Random Forest Classifier.
2. Separated out categorical columns, date columns and kept in separate df. Separated out the labels columns at this step.
3. Standardized all numerical columns data using Standard Scaler from sklearn.
4. Joined back the dropped categorical columns.
5. One-hot encoded all categorical data.

## Model building and training
1. Made k-fold cross-validation/ train-test split preparation (k=20)
2. Used a Random Forest Classifier to fit the data and make predictions. Score metric used was Mean absolute error.
Formula for scoring used:
score = max(0, (100-mean_absolute_error(y_test, pred)))

3. Prepared the Test data, standardizing and one hot encoding the categorical values. 
4. Made predictions on the test dataset. Also saved the trained model in pickle file for later use.

