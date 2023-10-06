import streamlit as st
from predictor import *

st.title('Fake News Detector')

st.subheader('Enter the text and press Crtl+Enter to see if its a fake news or not.')

text = st.text_area("Enter the text to check if it's fake news:")

print(text)

if text:
    is_real = predict(text)

    if is_real[0] == 1:
        st.write('It is real!!!')

    elif is_real[0] == 0:
        st.write("Oh thats probably a fake news...")