import pandas as pd

url='https://drive.google.com/file/d/1igd3Zx9Vy8xgQElRxgqI3kvDd7ephrql/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df = pd.read_csv(dwn_url)
print(df.head())