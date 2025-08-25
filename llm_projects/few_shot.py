few_shots = [
  {
    "input": "calculate the average duration of each mode",
    "output": 
      "df = df.sort_values('t_stamp').reset_index(drop=True)\n"
      "df['_duration_next_s'] = (df['t_stamp'].shift(-1) - df['t_stamp']).dt.total_seconds()\n"
      "group_id = (df['mode'] != df['mode'].shift()).cumsum()\n"
      "group_durations = df.groupby(group_id)['_duration_next_s'].sum()\n"
      "group_modes = df.groupby(group_id)['mode'].first()\n"
      "grouped = pd.DataFrame({'mode': group_modes, 'duration_s': group_durations}).reset_index(drop=True)\n"
      "avg = grouped.groupby('mode', as_index=False)['duration_s'].mean()\n"
      "print(avg.to_string(index=False))"
  },
  {
    "input": "find mean time spent in each mode using 'mode' and 't_stamp'",
    "output":
      "df = df.sort_values('t_stamp').reset_index(drop=True)\n"
      "df['_duration_next_s'] = (df['t_stamp'].shift(-1) - df['t_stamp']).dt.total_seconds()\n"
      "group_id = (df['mode'] != df['mode'].shift()).cumsum()\n"
      "durations = df.groupby(group_id)['_duration_next_s'].sum()\n"
      "modes = df.groupby(group_id)['mode'].first()\n"
      "grouped = pd.DataFrame({'mode': modes, 'duration_s': durations}).reset_index(drop=True)\n"
      "avg = grouped.groupby('mode', as_index=False)['duration_s'].mean()\n"
      "print(avg)"
  },
  {
    "input": "average seconds per mode",
    "output":
      "df = df.sort_values('t_stamp').reset_index(drop=True)\n"
      "df['_duration_next_s'] = (df['t_stamp'].shift(-1) - df['t_stamp']).dt.total_seconds()\n"
      "group_id = (df['mode'] != df['mode'].shift()).cumsum()\n"
      "summary = df.groupby(group_id).agg(mode=('mode','first'), duration_s=('_duration_next_s','sum')).reset_index(drop=True)\n"
      "avg = summary.groupby('mode', as_index=False)['duration_s'].mean()\n"
      "print(avg)"
  }
]
