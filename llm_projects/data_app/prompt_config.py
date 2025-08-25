few_shots = [
    {
        "input": "Show summary of Salary column",
        "output": "print(df['Salary'].describe())"
    },
   {
        "input": "calculate the salary based on the gender and age of the employee",
        "output": "df = decide_salary(df, 'Gender', 'AgeCategory')"
    },
    {
       "input": "show summary of the data",
       "output": "print(df.describe())"
    },
    {
        "input": "Plot Gender distribution",
        "output": "fig = px.histogram(df, x='Gender')\nfig"
    },
    {
        "input": "Create a category column 'AgeCategory' based on Age: <30=young, else= old",
        "output": "df['AgeCategory'] = df['Age'].apply(lambda x: 'young' if x < 30 else 'old')\ndf"
    },
    {
        "input": "Show summary of rawph column",
        "output": "print(df['raw_pH'].describe())"
    },
    {
        "input": "plot Raw Conductivity distribution",
        "output": "fig = px.histogram(df, x='raw_conductivity')\nfig"
    },
    {
        "input": "scatter plot feed ph vs feed conductivity",
        "output": "fig = px.scatter(df, x='feed_pH_1', y='feed_conductivity')\nfig"
    },
    {
        "input": "show summary of FEED FLOW RATE column",
        "output": "print(df['feed_flow_rate'].describe())"
    },
    {
        "input": "create bar chart of flux column",
        "output": "fig = px.bar(df, x=df.index, y='flux')\nfig"
    },
    {
        "input": "summary of permeate salinity",
        "output": "print(df['permeate_salinity'].describe())"
    },
      {
        "input": "plot raw conductivity",
        "output": "fig = px.line(df, x='t_stamp', y='raw_conductivity')\nfig"
    },
    {
        "input": "plot feed flow rate",
        "output": "fig = px.line(df, x='t_stamp', y='feed_flow_rate')\nfig"
    },
    {
        "input": "plot flux over time",
        "output": "fig = px.line(df, x='t_stamp', y='flux')\nfig"
    },
    {
        "input": "plot permeate salinity",
        "output": "fig = px.line(df, x='t_stamp', y='permeate_salinity')\nfig"
    },
    {
        "input": "show summary of feed ph",
        "output": "print(df['feed_pH_1'].describe())"
    },
    {
        "input": "plot feed conductivity",
        "output": "fig = px.line(df, x='t_stamp', y='feed_conductivity')\nfig"
    },
    {
        "input": "scatter plot feed pressure vs perm pressure",
        "output": "fig = px.scatter(df, x='feed_pressure', y='perm_pressure')\nfig"
    },
    {
        "input": "bar chart of state column",
        "output": "fig = px.bar(df['state'].value_counts().reset_index(), x='index', y='state')\nfig"
    },
    {
        "input": "create a new column efficiency = perm_flow / feed_flow_rate",
        "output": "df['efficiency'] = df['perm_flow'] / df['feed_flow_rate']\ndf"
    },
    {
        "input": "plot efficiency",
        "output": "fig = px.line(df, x='t_stamp', y='efficiency')\nfig"
    },
    {
        "input": "calculate permeate salinity from perm conductivity",
        "output": """def salinity_from_conductivity(conductivity):
    return np.where(
        conductivity > 7630,
        8.01E-11 * np.exp(((-50.6458 - np.log(conductivity)) ** 2) / 112.484),
        7.70E-20 * np.exp(((-90.4756 - np.log(conductivity)) ** 2) / 188.8844)
    )

df["permeate_salinity"] = salinity_from_conductivity(df["perm_conductivity"])
df"""
    },
    {
        "input": "calculate raw salinity from feed conductivity",
        "output": """def salinity_from_conductivity(conductivity):
    return np.where(
        conductivity > 7630,
        8.01E-11 * np.exp(((-50.6458 - np.log(conductivity)) ** 2) / 112.484),
        7.70E-20 * np.exp(((-90.4756 - np.log(conductivity)) ** 2) / 188.8844)
    )

df["raw_salinity"] = salinity_from_conductivity(df["feed_conductivity"])
df"""
    },
    {
        "input": "calculate brine salinity from conc conductivity",
        "output": """def salinity_from_conductivity(conductivity):
    return np.where(
        conductivity > 7630,
        8.01E-11 * np.exp(((-50.6458 - np.log(conductivity)) ** 2) / 112.484),
        7.70E-20 * np.exp(((-90.4756 - np.log(conductivity)) ** 2) / 188.8844)
    )

df["brine_salinity"] = salinity_from_conductivity(df["conc_conductivity"])
df"""
    },
    {
        "input": "calculate feed salinity",
        "output": """df["feed_salinity"] = (
    (df["raw_salinity"] * df["perm_flow"]) +
    (df["brine_salinity"] * df["recirculation_flow"])
) / df["feed_flow_rate"]
df"""
    },
     {
        "input": "Plot perm_flow over time",
        "output": """x_col = 't_stamp'
df[x_col] = pd.to_datetime(df[x_col], errors="coerce").dt.tz_localize(None)
fig = px.line(df, x=x_col, y='perm_flow')"""
    },
    {
        "input": "Plot raw_pH against timestamp",
        "output": """x_col = 't_stamp'
df[x_col] = pd.to_datetime(df[x_col], errors="coerce").dt.tz_localize(None)
fig = px.line(df, x=x_col, y='raw_pH')"""
    },
    {
        "input": "Plot raw_conductivity vs time",
        "output": """x_col = 't_stamp'
df[x_col] = pd.to_datetime(df[x_col], errors="coerce").dt.tz_localize(None)
fig = px.line(df, x=x_col, y='raw_conductivity')"""
    },
    {
        "input": "Plot permeate flow vs time",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
fig = px.line(df, x='t_stamp', y='perm_flow', title="Permeate Flow vs Time")
fig.show()
"""
    },
    {
        "input": "Visualize conductivity over timestamp",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
fig = px.line(df, x='t_stamp', y='raw_conductivity', title="Conductivity vs Time")
fig.show()
"""
    },
    {
        "input": "Make a line chart of pH values against date",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
fig = px.line(df, x='t_stamp', y='raw_pH', title="Raw pH vs Time")
fig.show()
"""
    },
    {
        "input": "Convert t_stamp to datetime",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
"""
    },
    {
        "input": "Fix timestamp column for analysis",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
"""
    },
    {
        "input": "Make sure time_stamp is parsable",
        "output": """
df['t_stamp'] = pd.to_datetime(df['t_stamp'], errors='coerce', infer_datetime_format=True)
df = df.dropna(subset=['t_stamp'])
if pd.api.types.is_datetime64tz_dtype(df['t_stamp']):
    df['t_stamp'] = df['t_stamp'].dt.tz_localize(None)
"""
    },
    {
        "input": "Reset the t_stamp to have continuous 3-second intervals from the first timestamp",
        "output": """
start_time = df['t_stamp'].min()
df = df.reset_index(drop=True)
df['t_stamp'] = [start_time + pd.Timedelta(seconds=3*i) for i in range(len(df))]
"""
    },
    {
        "input": "Reindex time column to fixed step of 5 seconds",
        "output": """
start_time = df['t_stamp'].min()
df = df.reset_index(drop=True)
df['t_stamp'] = [start_time + pd.Timedelta(seconds=5*i) for i in range(len(df))]
"""
    },
    {
        "input": "Remove gaps and generate uniform timeline at 1 minute interval",
        "output": """
start_time = df['t_stamp'].min()
df = df.reset_index(drop=True)
df['t_stamp'] = [start_time + pd.Timedelta(minutes=1*i) for i in range(len(df))]
"""
    }
]

prefix = """ 
    You are a Python data assistant.
    Rules:
    - Only return clean runnable Python code.
    - DO NOT include Markdown, backticks, comments, explanations, or extra text.
    - DO NOT include any Markdown, backticks, or explanations
    - Respond ONLY with the code
    - if df is not present, do NOT attempt to load CSV; ask for path from user input to load CSV
    - Use pandas for data manipulations
    - Use plotly.express (px) for plotting
    - Always update 'df' in place
    - NEVER save CSV unless instructed
    - Only return clean runnable Python code
    - DO NOT include any Markdown, backticks, or explanations
    - Respond ONLY with the code
    - when user ask to calculate the salary based on the gender and age category, use the predefined function decide_salary(df, gender, age_category), look for columns name that has similar name like gender, age_group, age_category etc.
    - when user ask to show summary of the data, use df.describe() and print the result
    - If 'df' is not present, do NOT attempt to load CSV; wait for user input
    - For plotting: ALWAYS create Plotly figures with variable name 'fig' (e.g., fig = px.bar(df, x='column'))
    - DO NOT call fig.show() or print(fig) - just create the figure object
    - DO NOT return JSON or figure data - create the actual Plotly figure object
    - When plotting, use: fig = px.chart_type(df, x='column', y='column') and end with just 'fig'
    - Before plotting with Plotly, always check if the x-axis column looks like a datetime (e.g., contains 'time', 'date', 'stamp').
    - If yes, convert it using: df[column] = pd.to_datetime(df[column], errors="coerce")
    - You are working with a pandas DataFrame. When a user provides a column name, it might be formatted differently in the DataFrame—for example: lowercase (age), uppercase (AGE), title case (Age), or with underscores (first_name). Your task is to find the matching column in the DataFrame, regardless of these variations.
        - Check for exact matches first.
        - If no exact match, try case-insensitive match.
        - If still no match, try replacing spaces with underscores or vice versa.
        - If multiple columns match, return all possible matches.
        - If no match is found, politely inform the user and ask for clarification.
        - Example Instructions for LLM:
            - Input: "age" → matches "Age" in DataFrame
            - Input: "first name" → matches "first_name" in DataFrame
            - Input: "SALARY" → matches "Salary" or "salary"
    """

suffix = """Current conversation:
    {history}
    Human: {input}
    AI:"""