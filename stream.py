import datetime
import streamlit as st
import joblib
import pandas as pd
import altair as alt

page = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.pexels.com/photos/5481201/pexels-photo-5481201.jpeg?auto=compress&cs=tinysrgb&w=600");
background-size: cover;
}

[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page,unsafe_allow_html=True)

df = pd.read_csv("app.csv")

location_dict = {'Adelaide': 0, 'Albany': 1, 'Albury': 2, 'AliceSprings': 3, 'BadgerysCreek': 4, 'Ballarat': 5,
                 'Bendigo': 6, 'Brisbane': 7, 'Cairns': 8, 'Canberra': 9, 'Cobar': 10, 'CoffsHarbour': 11,
                 'Dartmoor': 12, 'Darwin': 13, 'GoldCoast': 14, 'Hobart': 15, 'Katherine': 16, 'Launceston': 17,
                 'Melbourne': 18, 'MelbourneAirport': 19, 'Mildura': 20, 'Moree': 21, 'MountGambier': 22,
                 'MountGinini': 23, 'Newcastle': 24, 'Nhil': 25, 'NorahHead': 26, 'NorfolkIsland': 27,
                 'Nuriootpa': 28, 'PearceRAAF': 29, 'Penrith': 30, 'Perth': 31, 'PerthAirport': 32, 'Portland': 33,
                 'Richmond': 34, 'Sale': 35, 'SalmonGums': 36, 'Sydney': 37, 'SydneyAirport': 38, 'Townsville': 39,
                 'Tuggeranong': 40, 'Uluru': 41, 'WaggaWagga': 42, 'Walpole': 43, 'Watsonia': 44, 'Williamtown': 45,
                 'Witchcliffe': 46, 'Wollongong': 47, 'Woomera': 48}

windgustdir_mapping = {'NNW': 0, 'NW': 1, 'WNW': 2, 'N': 3, 'W': 4, 'WSW': 5, 'NNE': 6, 'S': 7, 'SSW': 8, 'SW': 9, 'SSE': 10,
                           'NE': 11, 'SE': 12, 'ESE': 13, 'ENE': 14, 'E': 15}
winddir9am_mapping = {'NNW': 0, 'N': 1, 'NW': 2, 'NNE': 3, 'WNW': 4, 'W': 5, 'WSW': 6, 'SW': 7, 'SSW': 8, 'NE': 9, 'S': 10,
                          'SSE': 11, 'ENE': 12, 'SE': 13, 'ESE': 14, 'E': 15}
winddir3pm_mapping = {'NW': 0, 'NNW': 1, 'N': 2, 'WNW': 3, 'W': 4, 'NNE': 5, 'WSW': 6, 'SSW': 7, 'S': 8, 'SW': 9, 'SE': 10,
                          'NE': 11, 'SSE': 12, 'ENE': 13, 'E': 14, 'ESE': 15}
raintoday_mapping = {'Yes': 1, 'No': 0}

df['Location'] = df['Location'].map(location_dict)
df['WindGustDir'] = df['WindGustDir'].map(windgustdir_mapping)
df['WindDir9am'] = df['WindDir9am'].map(winddir9am_mapping)
df['WindDir3pm'] = df['WindDir3pm'].map(winddir3pm_mapping)
df['RainToday'] = df['RainToday'].map(raintoday_mapping)


def load_model(algorithm):
    if algorithm == "CATBOOST":
        model = joblib.load("./models/cat.pkl")
    elif algorithm == "LOGISTIC REGRESSION":
        model = joblib.load("./models/logreg.pkl")
    elif algorithm == "RANDOM FOREST":
        model = joblib.load("./models/rf.pkl")
    elif algorithm == "GAUSSIAN NAIVE BAYES":
        model = joblib.load("./models/gnb.pkl")
    elif algorithm == "XTREME GRADIENT BOOST":
        model = joblib.load("./models/xgb.pkl")
    elif algorithm == "K NEAREST NEIGHBOUR":
        model = joblib.load("./models/knn.pkl")
    else:
        model = None
    return model

def predict(Location, Date,model):
    locationIntCode = location_dict.get(Location, None)
    model_input = df[df['Location'] == locationIntCode].drop(columns=['RainTomorrow', 'Date'])
    prediction = model.predict(model_input)
    prob_rain = ((sum(1 for x in prediction if x == 1)) / prediction.shape[0]) * 100
    return prob_rain

def display_map(location):
    locationIntCode = location_dict.get(location, None)
    lat = df[df['Location'] == locationIntCode]['Latitude'].values[0]
    lon = df[df['Location'] == locationIntCode]['Longitude'].values[0]
    coordinates = pd.DataFrame({'Latitude': [lat], 'Longitude': [lon]})
    
    st.write("Map for selected location:")
    st.map(coordinates,latitude='Latitude', longitude='Longitude', zoom=11,size=100)


#def display_map2(location):
   #lat = df[df['Location'] == locationIntCode]['Latitude'].values[0]
    #lon = df[df['Location'] == locationIntCode]['Longitude'].values[0]
    #m = folium.Map(location=[lat ,lon], zoom_start=10, control_scale=True)
    #folium.TileLayer('openstreetmap').add_to(m)
    #folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',attr='Esri', name='Esri Topographic').add_to(m)
    #folium_static(m)

#def display_map3(location):
    #locationIntCode = location_dict.get(location, None)
    #lat = df[df['Location'] == locationIntCode]['Latitude'].values[0]
    #lon = df[df['Location'] == locationIntCode]['Longitude'].values[0]
    #m = folium.Map(location=[lat ,lon], zoom_start=10, control_scale=True)
    #folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',attr='Esri', name='Esri Topographic').add_to(m)
    #folium_static(m)


def generate_model_comparisons_chart():
    # Dictionary containing model names as keys and their accuracies and precisions as values
    model_metrics = {
        'CatBoost': {'accuracy': 0.8632, 'precision': 0.7574},
        'RF': {'accuracy': 0.8526, 'precision': 0.6829},
        'LogReg': {'accuracy': 0.7777, 'precision': 0.5931},
        'GNB': {'accuracy': 0.7463, 'precision': 0.5832},
        'KNN': {'accuracy': 0.7706, 'precision': 0.5912},
        'XGB': {'accuracy': 0.8580, 'precision': 0.7299},
    }

    # Extracting model names, accuracies, and precisions
    model_names = list(model_metrics.keys())
    accuracies = [model_metrics[model]['accuracy'] for model in model_names]
    precisions = [model_metrics[model]['precision'] for model in model_names]

    # Create DataFrame for plotting
    df_acc = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
    df_pre = pd.DataFrame({'Model': model_names, 'Precision': precisions})

    # Plotting accuracy and precision using Altair
    st.title('Model Comparisons')
    accuracy_chart = alt.Chart(df_acc).mark_line(color='green').encode(
        x='Model',
        y='Accuracy'
    ).properties(
        title='Accuracy and Precision Comparison',
        width=600,
        height=300
    )

    precision_chart = alt.Chart(df_pre).mark_line(color='red').encode(
        x='Model',
        y='Precision'
    )

    combined_chart = accuracy_chart + precision_chart
    st.altair_chart(combined_chart) 

def main():
    
    
    st.title("CLOUDBURST PREDICTION USING MACHINE LEARNING")

    selected_algorithm = st.selectbox("Select Algorithm", ["CATBOOST", "LOGISTIC REGRESSION", "RANDOM FOREST", "GAUSSIAN NAIVE BAYES", "XTREME GRADIENT BOOST", "K NEAREST NEIGHBOUR"])
    model = load_model(selected_algorithm)
    if model is not None:
        Location = st.selectbox("Select location:", list(location_dict.keys()))
        Date = st.date_input("Enter Date", datetime.date.today())  # Example default date
        if st.button("Predict"):
            prob_rain = predict(Location, Date, model)
        
            st.success(f'The predicted probability of cloudburst for {Location} on {Date} is: {prob_rain:.2f}%')
        if st.button("Display Map"):
            display_map(Location)
        if st.button("Model Comparisons"):
            generate_model_comparisons_chart()
        
        #if st.button("Show Satellite Map"):
        #display_map2(Location)
    
        #if st.button("Show Topo Map"):
        #display_map2(Location)'''

    
    else:
        st.write("Please select an algorithm")

if __name__ == "__main__":
    main()

