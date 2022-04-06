import glob
import pandas as pd

for csv_file in glob.glob('**/*.csv'):
    df = pd.read_csv(csv_file)
    if 'Participant Private ID' in df.columns:
        df=df.drop(columns=['Participant Private ID'])
        df.to_csv(csv_file.replace('.csv', '_anon.csv'))
        print(f'{csv_file} anonymized')