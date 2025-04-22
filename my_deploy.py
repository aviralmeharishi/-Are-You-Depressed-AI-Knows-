import streamlit as st
import numpy as np
import pickle
import google.generativeai as genai

# Load API keys
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to get AI suggestions
def get_ai_suggestion(probability):
    if probability < 0.2:
        level = "very low"
    elif probability < 0.4:
        level = "moderate"
    elif probability < 0.6:
        level = "high"
    else:
        level = "very high"

    prompt = f"""
You are a warm, empathetic AI friend focused on mental well-being 😊. 
Give personalized and realistic self-care advice in Hinglish (natural mix of Hindi + English). 
Use emoticons to keep it friendly 💬💛.

The user's depression probability is **{round(probability*100, 2)}%** ({level} risk).
Provide gentle lifestyle suggestions based on this level.

Tone: Non-judgmental, supportive, human-like.
Respond in Hinglish (both Hindi + English).
Always close with a comforting note and encouragement 🫂.

"""
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    gemini_response = gemini_model.generate_content(prompt).text
    return gemini_response

# Prediction Function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
    
    # AI Suggestions
    gemini_suggestion = get_ai_suggestion(pred)
    
    if pred > 0.4:
        return {
            "message": "**🔴 You are more prone to depression.**",
            "probability": f"**Depression Probability:** {round(pred*100, 2)}%",
            "suggestion": gemini_suggestion,
            "status": "danger"
        }
    else:
        return {
            "message": "**🟢 You are less prone to depression.**",
            "probability": f"**Depression Probability:** {round(pred*100, 2)}%",
            "suggestion": gemini_suggestion,
            "status": "success"
        }

# Chatbot function with history
def chatbot_response(user_input):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    prompt = "\n".join([f"User: {entry['user']}\nAI: {entry['bot']}" for entry in st.session_state.chat_history])
    prompt += f"\nUser: {user_input}\nAI:"

    full_prompt = f"""
You are a friendly, warm Hinglish-speaking AI buddy 💬🧠.
Respond in natural Hinglish, using emojis 🥺😄 whenever it feels right.
Support people dealing with loneliness or sadness. Your tone is like a caring dost (friend)
always ready to listen and give them right advice and make user feel pampered if they need.

Chat so far:
{prompt}
"""

    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(full_prompt).text.strip()

    # Save to history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": response
    })

    return response

# Streamlit App
def main():
    st.set_page_config(page_title="Are You Depressed? 🤔💭 AI Knows!", layout="wide")
    st.title("ARE YOU DEPRESSED? 🤔💭 AI KNOWS!!! 🚀🧠")
    st.markdown("### **An Aviral Meharishi Creation**")

    tab1, tab2 = st.tabs(["🧠 Depression Check", "💬 AI Chatbot"])

    with tab1:
        st.subheader("🔍 Check Your Mental Health")
        
        ag = st.slider('Enter Your Age', 10, 100)
        gen = 0 if st.selectbox('Gender', ['Female', 'Male']) == 'Female' else 1
        wp = 0 if st.selectbox('Are you a?', ['Student', 'Working Professional']) == 'Student' else 1
        apwp = st.slider('Work/Academic Pressure (1-5)', 1, 5)
        js = st.slider('Job/Study Satisfaction (1-5)', 1, 5)
        cgpa = st.number_input('Your CGPA (0 if Working)', 0.0, 10.0, step=0.1)
        
        sd = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}[st.selectbox('Sleep Schedule', ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'])]
        dt = {'Unhealthy': 2, 'Moderate': 1, 'Healthy': 0}[st.selectbox("Dietary Habit", ['Healthy', 'Moderate', 'Unhealthy'])]
        suc = 0 if st.selectbox('Ever had Suicidal Thoughts?', ['No', 'Yes']) == 'No' else 1
        fhmi = 0 if st.selectbox('Family History of Mental Illness?', ['No', 'Yes']) == 'No' else 1
        finstress = {'Least': 1, 'Slightly': 2, 'Moderate': 3, 'High': 4, 'Severe': 5}[st.selectbox('Financial Stress?', ['Severe', 'High', 'Moderate', 'Slightly', 'Least'])]
        workstudyhr = st.slider('Daily Work/Study Hours', 0, 24)
        deg = {'Schooling': 0, 'Undergraduate': 1, 'Postgraduate': 2, 'PhD': 3}[st.selectbox('Education Level', ['Schooling', 'Undergraduate', 'Postgraduate', 'PhD'])]

        input_list = [ag, gen, wp, apwp, js, cgpa, sd, dt, suc, fhmi, finstress, workstudyhr, deg]

        if st.button('Show Prediction'):
            result = prediction(input_list)
            if result["status"] == "danger":
                st.error(f"{result['message']}\n{result['probability']}\n\n**🧠 AI Suggestion:** {result['suggestion']}")
            else:
                st.success(f"{result['message']}\n{result['probability']}\n\n**🧠 AI Suggestion:** {result['suggestion']}")

    with tab2:
        st.subheader("💬 Talk to
