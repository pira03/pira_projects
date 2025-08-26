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
import pandas as pd
import math
import traceback
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()
key= os.getenv("OPENAI_API_KEY")

# Initialize GPT-5 Nano
llm = ChatOpenAI(openai_api_key=key, model_name="gpt-5-mini", temperature=0.2 )


main_agent = MainAgent(
    llm, 
    [salinity_calculator, processed_mode], 
    system_prompt,
    debug=True
    )
st.title("Plant Time Series Data Analysis")

st.write("This is a simple app to analyze plant data.")

st.write("""Please make sure your csv file has the following columns:
- time_stamp: should be a datetime style, the app will automatically convert it to the correct format.
- mode: if is float, ask agent to convert it to a integer.
""")


st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

# Load CSV from uploaded file
if uploaded_file is not None and "df" not in st.session_state:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df.rename(columns={'time_stamp': 't_stamp'}, inplace=True)
        df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['t_stamp'])
        if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
            df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
        st.session_state.df = df
        
        
        st.success(f"CSV loaded successfully! Shape: {df.shape}")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
if "df" in st.session_state:
    st.header("Ask Questions About Your Data")
    user_input = st.text_input("Enter your command or question:")

    if st.button("Run") and user_input.strip():
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        relevant_few_shots = main_agent.example_selector.select_examples({"input": user_input})

            # Show retrieved examples before running the LLM
        if relevant_few_shots:
            st.subheader("Retrieved Few-Shot Examples")
            for ex in relevant_few_shots:
                st.write(f"**Q:** {ex['input']}")
                st.code(ex['output'], language="python")

        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Question: {input}\nAnswer: {output}"
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=main_agent.example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
            prefix="Here are some examples of similar questions and their answers:",
            suffix="Question: {input}\nAnswer:"
        )

        prompt = few_shot_prompt.format(input=user_input)

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(prompt)
        ]
        try:
            # Create the initial state for LangGraph
            initial_state = {"messages": messages}
            ai_response = main_agent.graph_agent.invoke(initial_state, config=config)
            generated_code = ai_response['messages'][-1].content
            st.code(generated_code, language="python")

            # Safe execution namespace
            exec_namespace = {"df": st.session_state.get("df", None),
                            "salinity_calculator": salinity_calculator,
                            "processed_mode": processed_mode,
                            "pd": pd,
                            "math": math
                            }

            output_buffer = io.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = output_buffer

            try:
                exec(generated_code, exec_namespace)
            except Exception as e:
                st.error(f"Error executing code: {str(e)}")
                st.error(traceback.format_exc())

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

        except Exception as e:
            sys.stdout = sys.__stdout__  # Restore stdout in case of error
            st.error("Error during agent execution!")
            st.error(f"**Exception:** {str(e)}")
            st.error("**Traceback:**")
            st.text(traceback.format_exc())


if "df" in st.session_state and st.button("Save as CSV"):
    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name="data.csv", mime="text/csv")



    


