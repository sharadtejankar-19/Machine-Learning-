import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv("uber.csv")

print(df.head())

prof = ProfileReport(df)
prof.to_file(output_file = 'output.html')

