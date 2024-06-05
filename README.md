# Cloudburst_Prediction_Using_ML
## How to run the project
- Create a python Virtual environment 
1. Navigate outside of your main project directory using the command
```terminal
   cd path_of_your_project
```
2. now run these 2 commands to create and activate your virtual environment
```terminal
   For Windows:
   python -m venv venv       -> TO create a virtual env
   .\venv\Scripts\activate   -> To activate it

   For MAC_OS
   python3 -m venv venv       -> Create
   source venv/bin/activate   -> Activate
```
- Install the necessary libraries by running the following command:
```
    pip install -r requirements.txt
```


- Create folder named 'models' inside the project directory to store the model

- Run CB_models.ipynb or CB_models.py to develop models which will we stored inside models folder
- Run Gradio.py to run the project using Gradio interface
- Run stream.py to run the project using Streamlit. To run the streamlit inteface navigate to your project directory , activate virtual environment and run the following command in terminal:
 ``` streamlit run stream.py```

 You can use any of the following IDE's to run the project: 
 
 --> Anaconda with jupyter

 --> Pycharm

 -->VS code
 

## About the Project
These points summarize the project features and details

### Models used 
1. The project consists of 6 supervised classification algorithms listed below
```
- KNN
- LogReg
- Gaussian Naive Bayes
- Random Forest Classifer
- CatBoost Classifier
- XGBoost Classifier
```
2. All the models expects 23 input features and predicts 1 output feature.
```
INPUT FEATURES: Location	MinTemp	 MaxTemp	Rainfall	Evaporation	Sunshine	WindGustDir	WindGustSpeed	WindDir9am	WindDir3pm	WindSpeed9am	WindSpeed3pm	Humidity9am	Humidity3pm	Pressure9am	Pressure3pm	Cloud9am	Cloud3pm	Temp9am	Temp3pm	RainToday  Latitude   Longitude

OUTPUT FEATURES: RainTomorrow
```
### Interfaces used

#### Gradio
- It expects the user to input all 23 features in order to predict the output. 
- It uses the model wiht highest accruacy and precision ie Catboost. 
- The output would be message displaying either of the three 'Definitely a Cloudburst'/'Maybe a Cloudburst'/'Not a Cloudburst'
#### Streamlit

- It more user-friendly since it expects the user to input the 'Location' and 'Date' features only while others are calculated internally.
- Also has 'Show Map' button which displays the map.
- You can choose any of the 6 models to predict using the dropdown menu.
- The output is a message box showing the probability percentage of the event i.e. Occurence of a Cloudburst.
- Also has a'Models Comparison' button where you can compare the Accuracies and Precisions of all 6 models.


