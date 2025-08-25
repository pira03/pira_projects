import streamlit as st
from chatbot import MainAgent, salinity_calculator, processed_mode, load_csv_from_path, save_csv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from prompt import system_prompt
import os
import io
import sys

load_dotenv()
key= os.getenv("OPENAI_API_KEY")

# Initialize GPT-5 Nano
llm = ChatOpenAI(openai_api_key=key, model_name="gpt-5-mini", temperature=1 )


main_agent = MainAgent(
    llm, 
    [salinity_calculator, processed_mode], 
    system_prompt,
    debug=True
    )
st.title("Plant Data Analysis")

st.write("This is a simple app to analyze plant data.")

st.write("Please provide the path of the CSV file to get started.")
data_path = st.text_input("Enter the path of the CSV file:")



if "df" not in st.session_state and data_path.strip():
    try:    
        df, response = load_csv_from_path(data_path)
        st.info(response)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")

user_input = st.text_input("Enter your command ")

if st.button("Run") and user_input.strip():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_message = {"messages": [HumanMessage(user_input)]}
    ai_response = main_agent.graph_agent.invoke(user_message, config=config)
    generated_code = ai_response['messages'][-1].content
    st.code(generated_code, language="python")

    # Safe execution namespace
    exec_namespace = {"df": st.session_state.get("df", None)}

    output_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output_buffer

    exec(generated_code, exec_namespace)

    # Restore stdout
    sys.stdout = sys_stdout

    # Get captured output
    captured_output = output_buffer.getvalue().strip()

# Display captured output in a readable way
    if captured_output:
        st.subheader("Output")
        # For multi-line output, st.text or st.code
        st.code(captured_output)


    # Update session state
    if "df" in exec_namespace:
        st.session_state.df = exec_namespace["df"]

    # Display DataFrame
    if "df" in exec_namespace and exec_namespace["df"] is not None:
        st.subheader("DataFrame preview")
        st.dataframe(exec_namespace["df"].head(10))

    # Display Plotly figure
    if "fig" in exec_namespace:
        st.subheader("Plot")
        st.plotly_chart(exec_namespace["fig"], use_container_width=True)



    


