import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# https://www.kaggle.com/austinreese/craigslist-carstrucks-data?select=vehicles.csv

# df = pd.read_csv('data/vehicles.csv')
df = pd.DataFrame({'price': [-99, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})

q4 = df['price'].quantile(1.0)
q3 = df['price'].quantile(0.75)
q2 = df['price'].quantile(0.5)
q1 = df['price'].quantile(0.25)
print(q1, q2, q3, q4)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(lower, upper)

df_iqr = df.loc[(df['price'] > lower) & (df['price'] < upper)]
sns.boxplot(data=df_iqr, y='price')
plt.show()

df['price_zscore'] = zscore(df['price'])
df_zscore = df.loc[df['price_zscore'].abs() <= 3]
sns.boxplot(data=df_zscore, y='price')
plt.show()
