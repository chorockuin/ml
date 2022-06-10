import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# https://www.kaggle.com/austinreese/craigslist-carstrucks-data?select=vehicles.csv

df = pd.read_csv('data/vehicles.csv')

q3 = df['price'].quantile(0.75)
q1 = df['price'].quantile(0.25)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df_iqr = df.loc[(df['price'] > lower) & (df['price'] < upper)]
sns.boxplot(data=df_iqr, y='price')
plt.show()

df['price_zscore'] = zscore(df['price'])
df_zscore = df.loc[df['price_zscore'].abs() <= 3]
sns.boxplot(data=df_zscore, y='price')
plt.show()
