import streamlit as st
import numpy as np
import pickle
import openai
import google.generativeai as genai

# Load API keys
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to get AI suggestions
def get_ai_suggestion(probability):
    prompt = f"Provide general and personalised mental health tips for someone with {probability*100:.0f}% depression probability. in both languages English and Hindi"


    # Gemini 2 Flash
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # Gemini 2 Flash
    gemini_response = gemini_model.generate_content(prompt).text

    return gemini_response

# Prediction Function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]  # Get depression probability
    
    # AI Suggestions
   gemini_suggestion = get_ai_suggestion(pred)
    
    if pred > 0.4:
        return f"""
        **You are more prone to depression**  
        **Depression Probability:** {round(pred*100, 2)}% 
        **Gemini 2 Flash Suggestion:** {gemini_suggestion}
        """
    else:
        return f"""
        **You are less prone to depression**  
        **Depression Probability:** {round(pred*100, 2)}%  
        **Gemini 2 Flash Suggestion:** {gemini_suggestion}
        """

# Streamlit UI
def main():
    st.title('ARE YOU DEPRESSED? 🤔')

    ag = st.slider('Enter Your Age', 10, 100, 25)
    gen = 0 if st.selectbox('Gender', ['Female', 'Male']) == 'Female' else 1
    wp = 0 if st.selectbox('Are you a?', ['Student', 'Working Professional']) == 'Student' else 1
    apwp = st.slider('Work/Academic Pressure (1-5)', 1, 5, 3)
    js = st.slider('Job/Study Satisfaction (1-5)', 1, 5, 3)
    cgpa = st.number_input('Your CGPA (0 if Working)', 0.0, 10.0, step=0.1)
    
    sd = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}[st.selectbox('Sleep Schedule', ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])]
    dt = {'Unhealthy': 2, 'Moderate': 1, 'Healthy': 0}[st.selectbox("Dietary Habit", ['Healthy', 'Moderate', 'Unhealthy'])]
    suc = 0 if st.selectbox('Ever had Suicidal Thoughts?', ['No', 'Yes']) == 'No' else 1
    fhmi = 0 if st.selectbox('Family History of Mental Illness?', ['No', 'Yes']) == 'No' else 1
    finstress = {'Least': 1, 'Slightly': 2, 'Moderate': 3, 'High': 4, 'Severe': 5}[st.selectbox('Financial Stress?', ['Severe', 'High', 'Moderate', 'Slightly', 'Least'])]
    workstudyhr = st.slider('Daily Work/Study Hours', 0, 24, 8)
    deg = {'Schooling': 0, 'Undergraduate': 1, 'Postgraduate': 2, 'PhD': 3}[st.selectbox('Education Level', ['Schooling', 'Undergraduate', 'Postgraduate', 'PhD'])]

    input_list = [ag, gen, wp, apwp, js, cgpa, sd, dt, suc, fhmi, finstress, workstudyhr, deg]

    if st.button('Show Prediction'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()
