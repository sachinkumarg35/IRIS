import streamlit as st

st.title('Classification Algorithm')
st.write("""
#Heading1
Which one is the best?
""")

value = st.number_input('Enter a number', None,None)
st.write(f'The value entered is {value}')
c = st.sidebar.slider('c', 0.01, 10.0)
st.write('THE SLIDER VALUE is {c}')