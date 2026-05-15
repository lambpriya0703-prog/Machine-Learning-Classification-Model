import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.title('Hybrid Machine learning Project')
st.header('Select Dataset to predict value!!')
st.subheader('Project Summary:')

summary = '''
This project predicts outcomes using three datasets—Iris, Wine, and Tennis Play.

Iris: Classifies iris plant species.

Wine: Predicts wine quality.

Tennis Play: Predicts whether tennis can be played.
'''

st.write(summary)

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title('Select Project 🎯')

user_project_selection = st.sidebar.radio(
    'Project List:',
    ['Iris', 'Wine', 'Play Tennis']
)

wine_url = "https://ars.els-cdn.com/content/image/1-s2.0-S2589721721000222-gr1.jpg"

if user_project_selection == 'Iris':
    st.sidebar.image('iris bg.png')

elif user_project_selection == 'Wine':
    st.sidebar.image(wine_url)

elif user_project_selection == 'Play Tennis':
    st.sidebar.image('play_tennis.jpg')

# =========================================================
# Dataset Load
# =========================================================

dataset_name = user_project_selection.lower().replace('play ', '')

# IMPORTANT
# make sure extension exists
temp_df = pd.read_csv(f'{dataset_name}.csv')

st.write(temp_df.sample(2))

# =========================================================
# User Input
# =========================================================

X_all_input = []

for col in temp_df.columns[:-1]:

    min_f = temp_df[col].min()
    max_f = temp_df[col].max()

    # Object / Categorical
    if temp_df[col].dtype == 'object':

        options = temp_df[col].unique()

        choice = st.sidebar.selectbox(
            f'Select {col}',
            options
        )

        X_all_input.append(choice)

    # Boolean
    elif str(temp_df[col].dtype) == 'bool':

        options = [True, False]

        choice = st.sidebar.selectbox(
            f'Select {col}',
            options
        )

        X_all_input.append(choice)

    # Numerical
    else:

        # FIXED HERE
        default_value = float(temp_df[col].sample(1).iloc[0])

        min_f = float(min_f)
        max_f = float(max_f)

        if min_f == max_f:
            max_f = max_f + 1

        choice = st.sidebar.slider(
            f'{col}',
            min_f,
            max_f,
            default_value
        )

        X_all_input.append(choice)

# =========================================================
# Final Input Data
# =========================================================

X_final_col = temp_df.columns[:-1]

X_input = pd.DataFrame(
    [X_all_input],
    columns=X_final_col
)

st.subheader('User Selected Choice:')
st.write(X_input)

# =========================================================
# Encoding for Tennis Dataset
# =========================================================

if user_project_selection == 'Play Tennis':

    X_input = pd.get_dummies(
        X_input,
        drop_first=True
    )

# =========================================================
# Load Model
# =========================================================

model_name = dataset_name + '_ml_brain.pkl'

with open(model_name, 'rb') as f:
    chatgpt_brain = pickle.load(f)

# =========================================================
# Prediction
# =========================================================

predicted_value = chatgpt_brain.predict(X_input)

final_predicted_value = predicted_value[0]

# =========================================================
# Target Labels
# =========================================================

iris_target_names = [
    'setosa',
    'versicolor',
    'virginica'
]

wine_target_names = [
    'Low Quality Wine',
    'Medium Quality Wine',
    'High Quality Wine'
]

tennis_target_names = [
    'Yes',
    'No'
]

# =========================================================
# Output
# =========================================================

if user_project_selection == 'Iris':

    st.image('iris bg.png')

    st.success(
        f'Predicted Flower: {iris_target_names[final_predicted_value]}'
    )

elif user_project_selection == 'Wine':

    st.image(wine_url)

    st.success(
        f'Predicted Wine Class: {wine_target_names[final_predicted_value]}'
    )

    if final_predicted_value == 0:
        st.image('wine_low.jpg', width=300)

    elif final_predicted_value == 1:
        st.image('wine_medium.jpg', width=300)

    else:
        st.image('wine_high.jpg', width=300)

elif user_project_selection == 'Play Tennis':

    st.image('play_tennis.jpg')

    result = tennis_target_names[final_predicted_value]

    st.warning(
        f'Can We Play Tennis? : {result}'
    )

    if result == 'Yes':
        st.image('tennis_yes.jpg', width=300)

    else:
        st.image('tennis_no.jpg', width=300)

# =========================================================
# Loading Animation
# =========================================================

with st.spinner('Thinking...'):
    time.sleep(2)

# =========================================================
# Footer
# =========================================================

st.markdown("Designed by **Priya**")
