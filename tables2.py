import numpy as np
import pandas as pd

pd.set_option("display.max.columns", 100)

from pylab import rcParams
rcParams['figure.figsize'] = 8, 5


import warnings


import matplotlib
matplotlib.use('Agg') # Явно указываем использовать графическое окно
import matplotlib.pyplot as plt 
import seaborn as sns # графики

warnings.filterwarnings("ignore") # реально нахуй надо

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
# df.info() # вывод информации про поля



# df['User_Score'] = df.User_Score.astype('float64') # смена типа
# df['Year_of_Release'] = df.Year_of_Release.astype('int64')
# df['User_Count'] = df.User_Count.astype('int64')
# df['Critic_Count'] = df.Critic_Count.astype('int64')


df = df.dropna() # убирает лишние записи, в которых нету части данных
print(df.shape) # возвращает размерность датафрейма



cols = [x for x in df.columns if 'Sales' in x] + ['Year_of_Release']
sales_df = df[cols]

ax = sales_df.groupby('Year_of_Release').sum().plot(figsize=(10, 6))

# 2. Добавляем заголовок и сетку (опционально)
plt.title('Sales by Year') # заголовок
plt.grid(True)

# 3. СОХРАНЯЕМ (укажите название файла и расширение)
plt.savefig('sales_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("График успешно сохранен в файл sales_plot.png")


cols = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')
plt.close()



corr_matrix = df.drop(['Other_Sales',  'Genre','User_Score','Developer','Rating','Publisher','Platform','Name'], axis=1).corr()
hm = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

