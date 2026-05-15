import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.title('Hybrid Machine Learning Project')
st.header('Select Dataset to Predict Value!!')
st.subheader('Project Summary:')

summary = '''
This project predicts outcomes using three datasets—Iris, Wine, and Tennis Play.

Iris: Classifies iris plant species based on flower measurements.

Wine: Predicts wine quality based on chemical properties.

Tennis Play: Predicts if a tennis match will be played based on weather conditions.
'''

st.write(summary)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title('Select Project 🎯')

user_project_selection = st.sidebar.radio(
    'Project List:',
    ['Iris', 'Wine', 'Play Tennis']
)

wine_url = 'https://ars.els-cdn.com/content/image/1-s2.0-S2589721721000222-gr1.jpg'

if user_project_selection == 'Iris':
    st.sidebar.image('iris bg.png')

elif user_project_selection == 'Wine':
    st.sidebar.image(wine_url)

elif user_project_selection == 'Play Tennis':
    st.sidebar.image('play_tennis.jpg')

# ==========================================================
# Read Dataset
# ==========================================================

file_name = user_project_selection.lower().replace('play ', '') + '.csv'

temp_df = pd.read_csv(file_name)

st.subheader("Sample Dataset")
st.write(temp_df.sample(2))

# ==========================================================
# User Input
# ==========================================================

X_all_input = []

feature_columns = temp_df.iloc[:, :-1].columns

for col in feature_columns:

    # Object datatype
    if temp_df[col].dtype == 'object':

        options = temp_df[col].unique()

        choice = st.sidebar.selectbox(
            f'Select {col}',
            options
        )

        X_all_input.append(choice)

    # Boolean datatype
    elif str(temp_df[col].dtype) == 'bool':

        options = [True, False]

        choice = st.sidebar.selectbox(
            f'Select {col}',
            options
        )

        X_all_input.append(choice)

    # Numeric datatype
    else:

        min_f = float(temp_df[col].min())
        max_f = float(temp_df[col].max())

        default_value = float(temp_df[col].sample(1).values[0])

        # if same value
        if min_f == max_f:
            max_f = max_f + 1

        choice = st.sidebar.slider(
            f'{col}',
            min_value=min_f,
            max_value=max_f,
            value=default_value
        )

        X_all_input.append(choice)

# ==========================================================
# Create Input DataFrame
# ==========================================================

X_input = pd.DataFrame(
    [X_all_input],
    columns=feature_columns
)

st.subheader('User Selected Values')
st.write(X_input)

# ==========================================================
# Encode Play Tennis
# ==========================================================

model_input = X_input.copy()

if user_project_selection == 'Play Tennis':

    encoded_df = pd.get_dummies(
        temp_df.iloc[:, :-1],
        drop_first=True,
        dtype=int
    )

    user_encoded = pd.get_dummies(
        X_input,
        drop_first=True,
        dtype=int
    )

    # Match columns
    user_encoded = user_encoded.reindex(
        columns=encoded_df.columns,
        fill_value=0
    )

    model_input = user_encoded

# ==========================================================
# Load Model
# ==========================================================

model_name = user_project_selection.lower().replace('play ', '')

final_model_name = model_name + '_ml_brain.pkl'

with open(final_model_name, 'rb') as f:
    chatgpt_brain = pickle.load(f)

# ==========================================================
# Prediction
# ==========================================================

predicted_value = chatgpt_brain.predict(model_input)

final_predicted_value = predicted_value[0]

# ==========================================================
# Target Labels
# ==========================================================

iris_target_names = ['setosa', 'versicolor', 'virginica']

wine_target_names = ['class_0', 'class_1', 'class_2']

tennis_target_names = ['Yes', 'No']

# ==========================================================
# Output Section
# ==========================================================

if user_project_selection == 'Iris':

    st.image('iris bg.png')

    target = iris_target_names

    ans_name = 'Predicted Flower is: '

elif user_project_selection == 'Wine':

    st.image(wine_url)

    target = wine_target_names

    ans_name = 'Predicted Wine Class: '

    if final_predicted_value == 0:

        class_name = 'Low Quality Wine'

        st.image('wine_low.jpg', width=300)

    elif final_predicted_value == 1:

        class_name = 'Medium Quality Wine'

        st.image('wine_medium.jpg', width=300)

    else:

        class_name = 'High Quality Wine'

        st.image('wine_high.jpg', width=300)

elif user_project_selection == 'Play Tennis':

    st.image('play_tennis.jpg')

    target = tennis_target_names

    ans_name = 'Can We Play Tennis?? : '

    if target[final_predicted_value] == 'Yes':

        st.image('tennis_yes.jpg', width=300)

    else:

        st.image('tennis_no.jpg', width=300)

# ==========================================================
# Loading Animation
# ==========================================================

with st.spinner('Thinking...'):
    time.sleep(2)

# ==========================================================
# Final Result
# ==========================================================

if user_project_selection == 'Wine':

    st.success(f'{ans_name} {class_name}')

else:

    st.warning(f'{ans_name} {target[final_predicted_value]}')

# ==========================================================
# Extra Iris Section
# ==========================================================

if user_project_selection == 'Iris':

    iris_arr = plt.imread('iris bg.png')

    image_arr = iris_arr.copy()

    setosa_arr = image_arr[:, 0:360, :]

    versi_arr = image_arr[:, 360:700, :]

    vergi_arr = image_arr[:, 700:, :]

    plt.imshow(setosa_arr)
    plt.axis('off')
    plt.savefig('setosa.jpeg')

    plt.imshow(versi_arr)
    plt.axis('off')
    plt.savefig('versicolor.jpeg')

    plt.imshow(vergi_arr)
    plt.axis('off')
    plt.savefig('virginica.jpeg')

    st.image(f'{target[final_predicted_value]}.jpeg')

# ==========================================================
# Footer
# ==========================================================

st.markdown("Designed by **Anshu**")
