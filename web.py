import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import *

# load model object 
api = joblib.load('./model/api.pkl')

# build model_dict
model_dict = {}
model_dict['V1'] = api

#title
st.title('PA control advice model')

# user select model
model_name = st.selectbox(
	'what model you want to use?',
	('V1',))
st.write('You selected',model_name)

# user input
user_input = st.number_input('input')
st.write('The user input is ',user_input)

# user press predict button
if st.button('predict'):
	model = model_dict[model_name]
	raw_advice = model.get_advice(user_input)
	output = model.get_critic_output(raw_advice)
	advice = model.pretty_advice(raw_advice)
	st.subheader('control advice')
	st.write(advice)
	st.subheader('predict output')
	st.write(output)
