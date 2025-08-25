# llm_backend.py
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI 
import os
from dotenv import load_dotenv
import io
import contextlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from prompt_config import prefix, suffix, few_shots

load_dotenv()

google_api_key = os.getenv("api_key")
openai_key= os.getenv("OPENAI_API_KEY")

# Initialize GPT-5 Nano


# -------------------------------
# Utilities
# -------------------------------
def decide_salary(df, gender_col: str, age_col: str) -> pd.DataFrame:
    def calc_salary(row):
        gender = str(row[gender_col]).lower()
        age = str(row[age_col]).lower()
        if gender == 'male' and age == 'young':
            return 75000
        elif gender == 'female' and age == 'young':
            return 100000
        else:
            return 50000

    df['Salary'] = df.apply(calc_salary, axis=1)
    return df
   

def clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = "\n".join(code.splitlines()[1:])
    if code.endswith("```"):
        code = "\n".join(code.splitlines()[:-1])
    return code

# -------------------------------
# Initialize LLM + ConversationChain
# -------------------------------
def init_conversation():
    google_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.2)
    openai_llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-5-mini", temperature=1 )

    
   

    memory = ConversationBufferMemory(return_messages=True)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create documents from few_shots examples for proper vectorstore creation
    documents = []
    
    for example in few_shots:
        # Create a document with the combined text
        text = f"{example['input']} {example['output']}"
        doc = Document(page_content=text, metadata=example)
        documents.append(doc)
    
    vectorstore = Chroma.from_documents(documents, embedding_model)

    # Example selector using semantic similarity
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,   # number of similar examples to retrieve
        input_keys=["input"]  # specify which field to use for similarity matching
    )
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Human: {input}\nAI:\n{output}"
    )

    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["history", "input"]
    )
    
    conversation = ConversationChain(llm=openai_llm, prompt=prompt, memory=memory, verbose=True)
    return conversation

# -------------------------------
# CSV Loading and Management
# -------------------------------
def load_csv_from_path(file_path: str) -> tuple[pd.DataFrame, str]:
    """
    Load CSV from file path and return DataFrame with success/error message
    """
    try:
        df = pd.read_csv(file_path.strip())
        df.columns = df.columns.str.strip()
        return df, f"CSV loaded from {file_path}"
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

def save_csv(df: pd.DataFrame, filename: str = "updated_data.csv") -> str:
    """
    Save DataFrame to CSV file
    """
    try:
        df.to_csv(filename, index=False)
        return f"CSV saved successfully as {filename}!"
    except Exception as e:
        return f"Error saving CSV: {str(e)}"

# -------------------------------
# Main Workflow Logic
# -------------------------------
def process_user_input(conversation, df: pd.DataFrame, user_input: str) -> dict:
    """
    Main function to process user input and return all results
    Returns a dictionary with all the necessary data for the UI
    """
    # Case A: CSV not loaded → check if input is a path
    if df is None:
        loaded_df, message = load_csv_from_path(user_input)
        if loaded_df is not None:
            # Successfully loaded CSV
            return {
                'df': loaded_df,
                'message': message,
                'code': None,
                'output': None,
                'plotly_figure': None,
                'response': None,
                'is_csv_loaded': True
            }
        else:
            # Not a valid path → let LLM respond
            _, response, _, _, _ = execute_user_input(conversation, None, user_input, decide_salary=decide_salary)
            return {
                'df': None,
                'message': None,
                'code': None,
                'output': None,
                'plotly_figure': None,
                'response': response,
                'is_csv_loaded': False
            }
    
    # Case B: CSV already loaded → normal LLM workflow
    else:
        df, response, code, output, plotly_figure = execute_user_input(conversation, df, user_input, decide_salary=decide_salary)
        return {
            'df': df,
            'message': None,
            'code': code,
            'output': output,
            'plotly_figure': plotly_figure,
            'response': response,
            'is_csv_loaded': True
        }

# -------------------------------
# Execute user input safely
# -------------------------------


def execute_user_input(conversation, df: pd.DataFrame, user_input: str, decide_salary=None):
    """
    Executes user input through the LLM conversation chain.
    Runs only in a controlled environment with allowed globals.
    """

    # Get LLM response (expected to be Python code)
    response = conversation.predict(input=user_input).strip()

    # If df is not loaded, just return response without execution
    if df is None:
        return None, response, None, None, None

    # Clean code (remove markdown fencing if present)
    code = clean_code(response)

    # Sandbox: only allow specific objects/functions
    allowed_globals = {
        "df": df,
        "pd": pd,
        "px": px,
    }
    if decide_salary is not None:
        allowed_globals["decide_salary"] = decide_salary
    
    stdout_capture = io.StringIO()
    plotly_figure = None

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, allowed_globals, allowed_globals)
        # run in controlled namespace
        df = allowed_globals.get("df", df)  # update df if modified
        output = stdout_capture.getvalue()
        
        # Check if a Plotly figure was created (look for 'fig' variable)
        if 'fig' in allowed_globals:
            plotly_figure = allowed_globals['fig']
            # Ensure the figure is properly configured for display
            if hasattr(plotly_figure, 'update_layout'):
                plotly_figure.update_layout(
                    title_x=0.5,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            # Debug: print figure type to help identify issues
            print(f"Figure type: {type(plotly_figure)}")
            print(f"Figure has to_dict: {hasattr(plotly_figure, 'to_dict')}")
            print(f"Figure has show: {hasattr(plotly_figure, 'show')}")
            
    except Exception as e:
        return df, response, None, f"Error in code execution: {e}", None

    return df, response, code, output, plotly_figure


