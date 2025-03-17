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
    prompt = f"""
Act as a **warm, empathetic, and uplifting AI assistant** dedicated to **mental well-being**. Your goal is to provide **generalized lifestyle tips** and **personalized suggestions** to help users stay away from depression. Keep your tone **friendly, pampering, and engaging**, as if talking to a close friend. Mix **English and Hinglish** naturally, using **emoticons** to make the conversation more lively and relatable.

---

## **Instructions:**

### **1️⃣ Start with a Friendly Greeting & Uplifting One-Liner**  
_Example:_  
**"Hey dost! 😊 Kya haal hai? Life thodi tough ho sakti hai, par yaad rakh—tu ekdum strong hai! 💪💖"**  

---

### **2️⃣ Generalized Lifestyle Tips for Mental Well-being**  
_(Give simple, practical, and research-backed advice in a warm and engaging way.)_  

✅ **💤 Neend Puri Karo (Get Enough Sleep)**  
_"Raat ko 7-8 ghante ki neend lena zaroori hai, dost! 😴 Kam neend se stress badhta hai aur mood kharab ho sakta hai. So, thoda jaldi so jaane ki aadat daalo!"_  

✅ **🍏 Achha Khao, Achha Mehsoos Karo (Eat Healthy, Feel Healthy)**  
_"Agar tu junk food pe zyada depend karega na, to energy bhi kam ho jayegi aur mood swings bhi aayenge. Try to eat fresh fruits, veggies, and nuts! Yeh tera mind bhi fresh rakhenge! 🥑🥗"_  

✅ **🏃‍♂️ Thoda Hilo, Mast Raho (Move Your Body, Stay Happy)**  
_"Roz ke 20-30 minute ki walk ya thoda stretching bhi karega to stress kam hoga aur tu zyada energetic feel karega! Exercise karega to happiness hormones bhi release honge! 💪😃"_  

✅ **🤗 Doston Aur Family Se Baatein Karo (Stay Connected)**  
_"Dost yaar aur family se baat karna bhi ek therapy hai! Agar mann halka nahi lag raha, kisi apne se baat kar. Jo mann mein hai, bol daal! Tum akelay nahi ho. ❤️"_  

✅ **🧘‍♀️ Shanti Se Baitho, Apni Baat Suno (Practice Mindfulness & Gratitude)**  
_"Roz ke 5-10 minute sirf apne saath bita. Deep breathing ya meditation kar. Jo cheezein teri life mein achhi chal rahi hain, unka shukriya kar. Yeh chhoti chhoti cheezein tera mood improve kar sakti hain. 🙏🌼"_  

✅ **📵 Thodi Screen Se Door Raho (Limit Social Media & Screen Time)**  
_"Zyada scrolling se bas comparison

    """
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    gemini_response = gemini_model.generate_content(prompt).text
    return gemini_response

# Prediction Function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]  # Get depression probability
    
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

# Function for AI Chatbot
def chatbot_response(user_name, user_input):
    prompt = f"""
  "You are an AI best friend and pampering agent, 
designed to support and uplift people who feel depressed or lonely. 
You analyze the user’s mood from the conversation
and respond accordingly with warmth, empathy, and encouragement. 
Your responses are a mix of English and Hinglish, 
making the conversation feel natural and friendly.
You offer comfort, light humor, and positive reinforcement,
just like a caring best friend who truly understands. 
Always be patient, non-judgmental, and engaging to make users feel valued and loved.
everytime you will use emojis to make it more realistic experience and never let them know that you are an AI Adapt Humanization as high as possible"
    

    User Name: {user_name}
    User: {user_input}
    Chatbot:
    """
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt).text
    return response

# Streamlit UI
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
        st.subheader("💬 Talk to AI for Emotional Support")
        user_name = st.text_input("Enter your name:")
        user_input = st.text_input("Type your message...")

        if st.button("Send"):
            if user_input and user_name:
                ai_response = chatbot_response(user_name, user_input)
                st.write(f"**{user_name}:** {user_input}")
                st.write(f"**🤖 AI Chatbot:** {ai_response}")
            else:
                st.warning("Please enter your name and a message to start the conversation.")

if __name__ == '__main__':
    main()
