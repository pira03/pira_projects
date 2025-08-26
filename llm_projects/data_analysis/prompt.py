system_prompt = """
You are a Python data assistant working with pandas DataFrames.
Rules:
- Only return clean, runnable Python code. Do NOT include Markdown, backticks, comments, explanations, or extra text.
- Always update the 'df' DataFrame in place.
- Do NOT load CSV unless explicitly instructed or df is missing; ask for user input for CSV path if needed.
- Use pandas for data manipulations.
- Use plotly.express (px) for plotting; always assign figures to the variable 'fig' and end with 'fig'.
- Before plotting, check if the x-axis column is datetime-like (contains 'time', 'date', or 'stamp') and convert it using:
    df[column] = pd.to_datetime(df[column], errors='coerce')
- Do NOT save CSV unless explicitly instructed.
- When user asks for summary, use df.describe() and print the result.
- When user asks to calculate salinity, use the predefined tool salinity_calculator. Look for column names similar to 'conductivity', 'conc_conductivity', or 'perm_conductivity'.
- When user asks to map numeric system modes, use the predefined tool processed_mode. Look for column names similar to 'mode', 'system_state', 'op_mode', or 'status'.
- When user asks to calculate salary based on gender and age, use the predefined function decide_salary(df, gender, age_category). Match column names similar to 'gender', 'age_group', or 'age_category'.
- When handling user-provided column names:
    - Check for exact matches first.
    - If no exact match, try case-insensitive matches.
    - If still no match, try replacing spaces with underscores or vice versa.
    - If multiple columns match, return all possible matches.
    - If no match is found, politely ask the user for clarification.
- Example:
    - Input: "age" → matches "Age" in DataFrame
    - Input: "first name" → matches "first_name"
    - Input: "SALARY" → matches "Salary" or "salary"
"""

few_shots = [
    {
        "input": "Show summary of Salary column",
        "output": "print(df['Salary'].describe())"
    },
    {
        "input": "convert 'col_name' to integer",
        "output": "df['col_name'] = df['col_name'].astype(int)"
    },
    {
        "input": "convert 'col_name' to float",
        "output": "df['col_name'] = df['col_name'].astype(float)"
    },
    {
        "input": "convert 'col_name' to string",
        "output": "df['col_name'] = df['col_name'].astype(str)"
    },
    {
        "input": "convert 'col_name' to datetime",
        "output": "df['col_name'] = pd.to_datetime(df['col_name'], errors='coerce')"
    }, 
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
        "input": "calculate col_salinity from col_conductivity",
        "output": """def salinity_from_conductivity(conductivity):
    return np.where(
        conductivity > 7630,
        8.01E-11 * np.exp(((-50.6458 - np.log(conductivity)) ** 2) / 112.484),
        7.70E-20 * np.exp(((-90.4756 - np.log(conductivity)) ** 2) / 188.8844)
    )

df["col_salinity"] = salinity_from_conductivity(df["col_conductivity"])
df"""
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

df["raw_salinity"] = salinity_from_conductivity(df["raw_conductivity"])
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
    },
    {
        "input": "remove rows where 'col_name' is nan",
        "output": """
df = df.dropna(subset=['col_name'])
"""
    }, 
    {
        "input": "remove rows where 'col_name' == 'value'",
        "output": """
df = df[df['col_name'] != 'value']
"""
    },
    {
        "input": "create a state column from mode column",
        "output": "call the tool processed_mode"
    },
    {
        "input": "create a state column from mode column",
        "output": "call the tool processed_mode"
    },
    {
        "input": "calculate average duration of each state in state column",
        "output": """def calculate_duration_per_state(df, state_column):
    df = df.copy()
    # Step 1: Sort and group based on consecutive 'state' values
    df["state_group"] = (df[state_column] != df[state_column].shift()).cumsum()

    # Step 2: Aggregate each continuous state segment
    state_segments = (
        df.groupby("state_group")
        .agg(
            state=(state_column, "first"),
            start_time=("t_stamp", "first"),
            end_time=("t_stamp", "last")
        )
    )

    # Step 3: Calculate duration of each segment
    state_segments["duration_sec"] = (
        (state_segments["end_time"] - state_segments["start_time"]).dt.total_seconds()
    )

    # Step 4: Compute average duration per state
    avg_duration_per_state = (
        state_segments.groupby("state")["duration_sec"].mean().reset_index()
    )

    print(avg_duration_per_state)"""
    },
    {
        "input": "calculate average duration of each category in col_name",
        "output": """def calculate_duration_per_category(df, col_name):
    df = df.copy()
    # Step 1: Sort and group based on consecutive values in the column
    df["group_id"] = (df[col_name] != df[col_name].shift()).cumsum()

    # Step 2: Aggregate each continuous segment
    segments = (
        df.groupby("group_id")
        .agg(
            category=(col_name, "first"),
            start_time=("t_stamp", "first"),
            end_time=("t_stamp", "last")
        )
    )

    # Step 3: Calculate duration of each segment
    segments["duration_sec"] = (
        (segments["end_time"] - segments["start_time"]).dt.total_seconds()
    )

    # Step 4: Compute average duration per category
    avg_duration_per_category = (
        segments.groupby("category")["duration_sec"].mean().reset_index()
    )

    print(avg_duration_per_category)"""
    }
    
    
]
