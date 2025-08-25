import streamlit as st
from helpers import init_conversation, process_user_input, save_csv

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Plant Data Analysis")

# 1️⃣ Init conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = init_conversation()

# 2️⃣ CSV not yet loaded → user must provide path
if "df" not in st.session_state:
    st.info("No CSV loaded yet. The assistant will ask for the path.")

# 3️⃣ User input
user_input = st.text_input("Enter your command or CSV path:")

if st.button("Run") and user_input.strip():
    # Process user input using the helper function
    result = process_user_input(st.session_state.conversation, st.session_state.get("df"), user_input)
    
    # Update session state if DataFrame was loaded or modified
    if result['df'] is not None:
        st.session_state.df = result['df']
    
    # Display results
    if result['message']:
        st.success(result['message'])
    
    if result['code']:
        st.code(result['code'])
    
    if result['output']:
        st.text("Output:")
        st.text(result['output'])
    
    # Display Plotly figure if created
    if result['plotly_figure'] is not None:
        st.plotly_chart(result['plotly_figure'], use_container_width=True)
    
    # Display DataFrame if available
    if result['df'] is not None:
        st.dataframe(result['df'].head())
    
    # Display LLM response
    if result['response']:
        st.write(result['response'])

# 4️⃣ Option to save
if "df" in st.session_state and st.button("Save CSV"):
    message = save_csv(st.session_state.df)
    st.success(message)
