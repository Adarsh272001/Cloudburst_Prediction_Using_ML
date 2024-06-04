# %%                              
import pickle
import pandas as pd
import gradio as gr
from datetime import datetime
import os

current_dir = os.path.dirname(__file__)
pkl_path = os.path.join(current_dir, 'models', 'cat.pkl')

# Load your machine learning model using pickle
with open(pkl_path, 'rb') as file:
    model = pickle.load(file)
print("Model Loaded")
csv_path = os.path.join(current_dir, 'weatherAUS.csv')

# Read the CSV file
df = pd.read_csv(csv_path)


# Extracting unique values for some specific columns and displaying it as a dropdown in the interface
unique_locations = df['Location'].dropna().unique().tolist()
unique_wind_dir_9am = df['WindDir9am'].dropna().unique().tolist()
unique_wind_dir_3pm = df['WindDir3pm'].dropna().unique().tolist()
unique_wind_gust_dir = df['WindGustDir'].dropna().unique().tolist()
unique_raintoday = df['RainToday'].dropna().unique().tolist()

def predict(date, mintemp, maxtemp, rainfall, evaporation, sunshine, windgustspeed, windspeed9am, windspeed3pm,
            humidity9am, humidity3pm, pressure9am, pressure3pm, temp9am, temp3pm, cloud9am, cloud3pm, location,
            winddir9am, winddir3pm, windgustdir, raintoday):

    # Convert date to day and month
    parsed_date = datetime.strptime(date, "%Y-%m-%d")
    day = parsed_date.day
    month = parsed_date.month
    
    location_mapping = {'Portland': 1, 'Cairns': 2, 'Walpole': 3, 'Dartmoor': 4, 'MountGambier': 5, 'NorfolkIsland': 6, 
                        'Albany': 7, 'Witchcliffe': 8, 'CoffsHarbour': 9, 'Sydney': 10, 'Darwin': 11, 'MountGinini': 12, 
                        'NorahHead': 13, 'Ballarat': 14, 'GoldCoast': 15, 'SydneyAirport': 16, 'Hobart': 17, 'Watsonia': 18, 
                        'Newcastle': 19, 'Wollongong': 20, 'Brisbane': 21, 'Williamtown': 22, 'Launceston': 23, 'Adelaide': 24, 
                        'MelbourneAirport': 25, 'Perth': 26, 'Sale': 27, 'Melbourne': 28, 'Canberra': 29, 'Albury': 30, 'Penrith': 31, 
                        'Nuriootpa': 32, 'BadgerysCreek': 33, 'Tuggeranong': 34, 'PerthAirport': 35, 'Bendigo': 36, 'Richmond': 37, 
                        'WaggaWagga': 38, 'Townsville': 39, 'PearceRAAF': 40, 'SalmonGums': 41, 'Moree': 42, 'Cobar': 43, 'Mildura': 44, 
                        'Katherine': 45, 'AliceSprings': 46, 'Nhil': 47, 'Woomera': 48, 'Uluru': 49}
    
    windgustdir_mapping = {'NNW': 0, 'NW': 1, 'WNW': 2, 'N': 3, 'W': 4, 'WSW': 5, 'NNE': 6, 'S': 7, 'SSW': 8, 'SW': 9, 'SSE': 10,
                           'NE': 11, 'SE': 12, 'ESE': 13, 'ENE': 14, 'E': 15}
    winddir9am_mapping = {'NNW': 0, 'N': 1, 'NW': 2, 'NNE': 3, 'WNW': 4, 'W': 5, 'WSW': 6, 'SW': 7, 'SSW': 8, 'NE': 9, 'S': 10,
                          'SSE': 11, 'ENE': 12, 'SE': 13, 'ESE': 14, 'E': 15}
    winddir3pm_mapping = {'NW': 0, 'NNW': 1, 'N': 2, 'WNW': 3, 'W': 4, 'NNE': 5, 'WSW': 6, 'SSW': 7, 'S': 8, 'SW': 9, 'SE': 10,
                          'NE': 11, 'SSE': 12, 'ENE': 13, 'E': 14, 'ESE': 15}
    raintoday_mapping = {'Yes': 1, 'No': 0}

    encoded_raintoday = raintoday_mapping.get(raintoday, -1)
    encoded_location = location_mapping.get(location, -1)
    encoded_windgustdir = windgustdir_mapping.get(windgustdir, -1)
    encoded_winddir9am = winddir9am_mapping.get(winddir9am, -1)
    encoded_winddir3pm = winddir3pm_mapping.get(winddir3pm, -1)

    # Check if any mapping failed
    if -1 in (encoded_location, encoded_windgustdir, encoded_winddir9am, encoded_winddir3pm):
        return "Unknown value for categorical feature"

    input_lst = [encoded_location, float(mintemp), float(maxtemp), float(rainfall), float(evaporation), float(sunshine),
                encoded_windgustdir, float(windgustspeed), encoded_winddir9am, encoded_winddir3pm, float(windspeed9am),
                 float(windspeed3pm), float(humidity9am), float(humidity3pm), float(pressure9am), float(pressure3pm),
                 float(cloud9am), float(cloud3pm), float(temp9am), float(temp3pm),  encoded_raintoday, float(month), float(day)]

    pred = model.predict([input_lst])

    if pred[0] == 0:
        return "No Cloubrust"
    else:
        return "Maybe a Cloudburst"


# Create Gradio interface without using HTML file
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter Date (YYYY-MM-DD)"),  # date
        gr.Number(),  # mintemp
        gr.Number(),  # maxtemp
        gr.Number(),  # rainfall
        gr.Number(),  # mevaporation
        gr.Number(),  # sunshine
        gr.Number(),  # windgustspeed
        gr.Number(),  # windspeed(9am)
        gr.Number(),  # windspeed(3pm)
        gr.Number(),  # humidity(9am)
        gr.Number(),  # humidity(3pm)
        gr.Number(),  # pressure(9am)
        gr.Number(),  # pressure(3pm)
        gr.Number(),  # temp(9am)
        gr.Number(),  # temp(3pm)
        gr.Number(),  # cloud(9am)
        gr.Number(),  # cloud(3pm)
        gr.Dropdown(unique_locations, label="Location"),  # location
        gr.Dropdown(unique_wind_dir_9am, label="Wind Direction at 9am"),  # winddir(9am)
        gr.Dropdown(unique_wind_dir_3pm, label="Wind Direction at 3pm"),  # winddir(3pm)
        gr.Dropdown(unique_wind_gust_dir, label="Wind Gust Direction"),  # windgustdir
        gr.Dropdown(unique_raintoday, label='Rain Today')  # raintoday
    ],
    outputs=gr.Textbox(),
)

# Launch the Gradio interface
iface.launch(share=True)


# %%


# %%


# %%
