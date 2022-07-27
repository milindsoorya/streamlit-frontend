import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main(
        background-color: #f5f5f5;
    )
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache
def get_data(filename):
    taxi_data = pd.read_parquet(filename)

    return taxi_data


with header:
    st.title('Welcome to the Machine Learning Project')
    st.text('In this project, we will be using the NYC Taxi Fare Data Set to predict the fare amount of a taxi ride.')

with dataset:
    st.header('NYC taxi dataset')
    st.text('Found the dataset here - ')
    st.markdown(
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet')

    taxi_data = get_data('./data/yellow_tripdata_2022-01.parquet')
    # st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pulocation_dist = pd.DataFrame(
        taxi_data['PULocationID'].value_counts().head(5))
    st.bar_chart(pulocation_dist)

with features:
    st.header('The features I created')

    st.markdown(
        '* **first feature: I created this feature because of this... I calculated it using this lofic...**')
    st.markdown(
        '* **second feature: I created this feature because of this... I calculated it using this lofic...**')


with model_training:
    st.header('Time to train the model')
    st.text(
        'Here you can change the hyperparmeters of the model and see how it performs')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("What should be the max_depth of the model?",
                               min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox("How many trees should be there?", options=[
                                     100, 200, 300, "No limit"], index=0)

    if n_estimators == "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators)

    sel_col.text("Here are the available features:")
    sel_col.table(taxi_data.columns)

    input_feature = sel_col.text_input(
        "Which feature should be used as the input feature?", value='PULocationID')

    X = taxi_data[[input_feature]].values
    y = taxi_data[['trip_distance']].values

    regr.fit(X, y.ravel())
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_absolute_percentage_error(y, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))
