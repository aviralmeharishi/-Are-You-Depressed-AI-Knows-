import streamlit as st
import numpy as np
import pickle
import google.generativeai as genai

# Load API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load ML model
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# AI suggestion for depression result
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
You are a warm, empathetic AI friend focused on mental well-being act like a well certified psychologist. 
Give personalized and realistic self-care advice in English and Hindi. 
Use friendly emojis (💬💛) and comforting tone.

The user's depression probability is **{round(probability*100, 2)}%** ({level} risk).
Provide 2-3 gentle, practical lifestyle suggestions for this level.

Tone: Supportive, non-judgmental as well as professional 
Close with a sweet encouragement and positive energy 🫂.
"""
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    gemini_response = gemini_model.generate_content(prompt).text
    return gemini_response

# Model prediction function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]
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

# Chatbot function with optional chat history
def chatbot_response(user_input, chat_enabled, history_list):
    base_prompt = """
You are a caring, supportive and cheerful AI friend. Respond in a warm, soft and pampering and decent tone.
Help people feel better, pampered a little, and emotionally supported always reply with an engaging long and catchy text .

Response format:

Response it in Hinglish (mix of Hindi and English) but dont use Jaanu ike creepy words in hindi .
Use friendly emojis occasionally (😊, 💛, 🌸) but don't overdo it.

Avoid sounding robotic. Talk like a close dost.
"""

    if chat_enabled and history_list:
        history_prompt = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in history_list])
        prompt = f"{base_prompt}\n\nPrevious Conversation:\n{history_prompt}\n\nUser: {user_input}\nAI:"
    else:
        prompt = f"{base_prompt}\nUser: {user_input}\nAI:"

    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt).text.strip()

    if chat_enabled:
        history_list.append({"user": user_input, "ai": response})

    return response

# Streamlit App UI
def main():
    st.set_page_config(page_title="Are You Depressed? 🤔💭 AI Knows!", layout="wide")
    st.title("ARE YOU DEPRESSED? 🤔💭 AI KNOWS!!! 🚀🧠")
    st.markdown("### **An Aviral Meharishi Creation**")

    tab1, tab2 = st.tabs(["🧠 Depression Check", "💬 AI Chatbot"])

    # Depression Checker Tab
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

    # Chatbot Tab
    with tab2:
        st.subheader("💬 Talk to AI for Emotional Support")

        chat_enabled = st.sidebar.toggle("Enable Chat History", value=False)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Type your message here... Chinta ki Kya hai Baat Jab Mai Hu Sath")

        if st.button("Send"):
            if user_input:
                response = chatbot_response(user_input, chat_enabled, st.session_state.chat_history)
                st.markdown(f"**🤖 AI:** {response}")
            else:
                st.warning("Please write something to start the conversation.")

# Run the app
if __name__ == '__main__':
    main()
