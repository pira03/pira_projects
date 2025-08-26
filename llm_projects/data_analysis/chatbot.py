
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
import plotly.express as px
import pandas as pd
from langchain_core.tools import tool
import math
import numpy as np
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
import json
import uuid
from langchain_community.vectorstores import FAISS
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.docstore.document import Document  
from langchain_openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings 
from prompt import few_shots


# load_dotenv()
# key= os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(openai_api_key=key, model_name="gpt-5-mini", temperature=1 )



@tool
def salinity_calculator(conductivity: float) -> float:
    """
    Calculate the brine salinity from a single conductivity measurement.

    This function estimates the salinity of a solution (e.g., feed water or concentrate)
    based on its electrical conductivity, which is commonly measured in µS/cm in this dataset. 
    Different exponential formulas are applied depending on whether the conductivity exceeds 
    a threshold of 7630 µS/cm.

    Parameters
    ----------
    conductivity : float
        Electrical conductivity of the solution (µS/cm).  
        Typically sourced from columns like 'conc_conductivity', 'perm_conductivity', or 'feed_conductivity'.

    Returns
    -------
    float
        Estimated brine salinity corresponding to the given conductivity value.

    Notes
    -----
    - For conductivity > 7630 µS/cm, a high-conductivity model is applied.
    - For conductivity <= 7630 µS/cm, a low-conductivity model is applied.
    - Non-numeric or missing values should be handled before calling this function.
    - Designed to work with individual float values (not pandas Series).
    """
    if conductivity > 7630:
        return 8.01e-11 * math.exp(((-50.6458 - math.log(conductivity)) ** 2) / 112.484)
    else:
        return 7.70e-20 * math.exp(((-90.4756 - math.log(conductivity)) ** 2) / 188.8844)


@tool
def processed_mode(mode: int) -> str:
    """
    Map a numeric system mode code to a human-readable operational state.

    The dataset contains a 'mode' column representing system operation codes. 
    This function converts these numeric codes into descriptive states:
    'produce', 'flush', or 'offline'.

    Parameters
    ----------
    mode : int
        Numeric mode code from the 'mode' column in the dataset.

    Returns
    -------
    str
        Operational state corresponding to the code:
        - "produce" for mode codes 401–406  
        - "flush" for mode codes 431–465  
        - "offline" for all other codes

    Notes
    -----
    - Only integer values are valid; non-integer entries should be converted before calling.
    - Useful for transforming the 'mode' column into human-readable state labels 
      for analysis or visualization.
    """
    if 401 <= mode <= 406:
        return "produce"
    elif 431 <= mode <= 465:
        return "flush"
    else:
        return "offline"




class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class MainAgent:
    def __init__(self, model, tools, system_prompt=None, debug=False):

        self.system_prompt = system_prompt
        self.debug = debug

        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # build vectorstore and example selector
        documents = []
    
        for example in few_shots:
            # Create a document with the combined text
            text = f"{example['input']} {example['output']}"
            doc = Document(page_content=text, metadata=example)
            documents.append(doc)

        self.vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=few_shots,           # your list of few-shot examples
            embeddings=self.embeddings,
            input_keys=["input"],
            k=3,
            vectorstore_cls=FAISS         # optional
        )
        

        # build state graph
        graph_agent = StateGraph(AgentState)
        graph_agent.add_node("agent", self.call_agent)
        graph_agent.add_node("tools", self.call_tools)
        graph_agent.add_conditional_edges(
            "agent",
            self.is_tool_call,
            {True: "tools", False: END}
        )
        graph_agent.add_edge("tools", "agent")
        graph_agent.set_entry_point("agent")


        self.memory = MemorySaver()
        self.graph_agent = graph_agent.compile(checkpointer=self.memory)

        # setup the tools
        self.tools = {tool.name: tool for tool in tools}
        callable_tools = [t.func if hasattr(t, "func") else t for t in tools]

        if self.debug:
            print("\nTools loaded:", list(self.tools.keys()))

        # setup the model with tools bound
        self.model = model.bind_tools(callable_tools)

    def call_agent(self, state: AgentState) -> AgentState:
        messages = state["messages"]

        # prepend system prompt once
        if self.system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = self.model.invoke(messages)
        if self.debug:
            print(f"\nAgent response: {response}")

        return {"messages": [response]}

    def is_tool_call(self, state: AgentState) -> bool:
        last_message = state["messages"][-1]
        return bool(last_message.tool_calls)

    def call_tools(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for tool in tool_calls:
            if tool["name"] not in self.tools:
                result = "Invalid tool found. Please retry"
            else:
                args = tool["args"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        if self.debug:
                            print(f"Failed to parse args for tool {tool['name']}: {args}")
                result = self.tools[tool["name"]].invoke(args)

            results.append(
                ToolMessage(
                    tool_call_id=tool["id"],
                    name=tool["name"],
                    content=str(result),
                )
            )

        if self.debug:
            print(f"\nTools returned: {results}")
        return {"messages": results}


def run_agent(main_agent, user_input): 

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    
    # Format the user message
    user_message = HumanMessage(user_input)
    
    # Get response from the agent
    ai_response = main_agent.graph_agent.invoke(user_message, config=config)
    
    # Get the code generated by the agent
    generated_code = ai_response['messages'][-1].content
    print(f"\nAGENT generated code:\n{generated_code}")
    
     # Execute in a separate namespace
    exec_namespace = {}
    exec(generated_code, exec_namespace)

    # Show DataFrame if created
    if 'df' in exec_namespace:
        print("\nAGENT sees the DataFrame:")
        print(exec_namespace['df'].head(10))

    if 'fig' in exec_namespace:
        exec_namespace['fig'].show()



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
        df.rename(columns={'time_stamp': 't_stamp'}, inplace=True)
        df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['t_stamp'])
        if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
            df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
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