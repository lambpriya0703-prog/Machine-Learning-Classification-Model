import streamlit as st
import time
import pandas as pd
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


st.title('Hybrid Machine learning Project')
st.header('Select Dataset to predict value!!')
st.subheader('Project Summary:')

summary = '''
This project predicts outcomes using three datasets—Iris, Wine, and Tennis Play. Users can select features to generate predictions for classification tasks:

Iris: Classifies iris plant species based on flower measurements.

Wine: Predicts wine quality based on chemical properties.

Tennis Play: Predicts if a tennis match will be played based on weather conditions.

The system uses machine learning models to provide insights based on the selected features.'''

st.write(summary)


st.sidebar.title('Select Project 🎯 ')
user_project_selection = st.sidebar.radio('Project List: ',['Iris','Wine','Play Tennis'])
# st.sidebar.write(user_project_selection)
# st.write(time.asctime())

wine_url ='''https://ars.els-cdn.com/content/image/1-s2.0-S2589721721000222-gr1.jpg'''


if user_project_selection == 'Iris':
    st.sidebar.image('iris bg.png')
elif user_project_selection == 'Wine':
    st.sidebar.image(wine_url)
elif user_project_selection == 'Play Tennis':
    st.sidebar.image('play_tennis.jpg')

temp_df = pd.read_csv(user_project_selection.lower().replace('play ',''))
st.write(temp_df.sample(2))





np.random.seed(23)

X_all_input = []

for i in temp_df.iloc[:,:-1]:
    min_f, max_f = temp_df[i].agg(['min','max']).values
    if str(temp_df[i].agg(['min','max']).dtype) == 'object':
        options = temp_df[i].unique()
        choice = st.sidebar.selectbox(f'select {i} value',options)
        st.sidebar.write(f"You selected :{choice}")
        X_all_input.append(choice)

    else:
        if min_f == max_f:
            max_f == (max_f + 1)
        else:
            if str(temp_df[i].dtype) == 'bool':
                options = [True,False]
                choice = st.sidebar.selectbox(f'select {i} value',options)
                X_all_input.append(choice)
                st.sidebar.write(f"You selected :{choice}")
            else:
                choice = st.sidebar.slider(f'{i}',min_f,max_f,temp_df[i].sample(1).values[0])
                X_all_input.append(choice)


X_final_col = temp_df.iloc[:,:-1].columns
final_X = [X_all_input]

X_input = pd.DataFrame(final_X,columns = X_final_col)

st.subheader('User Selected Choice:')
st.write(X_input)

if user_project_selection == 'Play Tennis':
    final_X = pd.get_dummies(temp_df.iloc[:,:-1],drop_first=True,dtype = int)


#===============================================================
# Model Call
model_name = user_project_selection.lower().replace('play ','')
final_model_name = model_name + '_ml_brain.pkl'
with open(final_model_name,'rb') as f:
    chatgpt_brain = pickle.load(f)

predicted_value = chatgpt_brain.predict(final_X)
final_predicted_value = predicted_value[0]


iris_target_names = ['setosa', 'versicolor', 'virginica']
wine_target_names = ['class_0','class_1','class_2']
tennis_target_names = ['Yes','No']

if user_project_selection == 'Iris':
    st.image('iris bg.png')
    target = iris_target_names
    ans_name = 'Predicted Flower is : '
    
elif user_project_selection == 'Wine':
    st.image(wine_url)
    target = wine_target_names
    ans_name = 'Predicted Wine Class: '
    class_name = ''
    if final_predicted_value == 0:
        class_name = 'Low Quality Wine'
    elif final_predicted_value == 1:
        class_name = 'Medium Quality Wine'
    else: 
        class_name = 'High Quality Wine'
    
elif user_project_selection == 'Play Tennis':
    st.image('play_tennis.jpg')
    target = tennis_target_names
    ans_name = 'Can We Play Tennis ??: '

#if target[final_predicted_value] == 'No':
 #   st.warning(f'{ans_name} {target[final_predicted_value]}')
#else:

with st.spinner('Thinking...'):
 time.sleep(3)
if user_project_selection == 'Wine':
    st.success(f'{ans_name} {class_name}')
else:
    st.warning(f'{ans_name} {target[final_predicted_value]}')
    
if user_project_selection == 'Iris' :
    iris_arr = plt.imread('iris bg.png')
    image_arr = iris_arr.copy()
    setosa_arr = image_arr[:,0:360,:]
    versi_arr = image_arr[:,360:700,:]
    vergi_arr = image_arr[:,700:,:]
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
    

  











