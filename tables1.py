import numpy as np
import pandas as pd

pd.set_option("display.max.columns", 100)

from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

import pandas as pd
import warnings

import matplotlib.pyplot as plt# для красивых табличек
import seaborn as sns # графики

warnings.filterwarnings("ignore")# реально нахуй надо

DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
data = pd.read_csv(DATA_URL + "adult.data.csv") #пандас читает и преобразует файл в читаемый формат
data.head()

print(data["sex"].value_counts(),'\n')

print(data[data["sex"] == "Female"]["age"].mean(),'\n') #среднее значение по полю age, где sex = Female

ages1 = data[data["salary"] == ">50K"]["age"]
ages2 = data[data["salary"] == "<=50K"]["age"]
print(
    "The average age of the rich: {0} +- {1} years, poor - {2} +- {3} years.".format(
        round(ages1.mean()),
        round(ages1.std(), 1),
        round(ages2.mean()),
        round(ages2.std(), 1),
    )
)



