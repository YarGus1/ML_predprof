from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
# %matplotlib inline
import seaborn as sns
from matplotlib import pyplot as plt


'''
plt.rcParams['figure.figsize'] = (6,4)
xx = np.linspace(0,1,50)
plt.plot(xx, [2 * x * (1-x) for x in xx], label='gini')
plt.plot(xx, [4 * x * (1-x) for x in xx], label='2*gini')
plt.plot(xx, [-x * np.log2(x) - (1-x) * np.log2(1 - x)  for x in xx], label='entropy')
plt.plot(xx, [1 - max(x, 1-x) for x in xx], label='missclass')
plt.plot(xx, [2 - 2 * max(x, 1-x) for x in xx], label='2*missclass')
plt.xlabel('p+')
plt.ylabel('criterion')
plt.title('Критерии качества как функции от p+ (бинарная классификация)')
plt.legend();
plt.savefig('entropy.png', dpi=300, bbox_inches='tight')


'''

# def build(L):
#     create node t
#     if the stopping criterion is True:
#         assign a predictive model to t
#     else:
#         Find the best binary split L = L_left + L_right
#         t.left = build(L_left)
#         t.right = build(L_right)
#     return t  




# Рандомная генерация данных


plt.rcParams['figure.figsize'] = (10,8)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
plt.plot(range(-2,5), range(4,-3,-1))
plt.savefig('entropy.png', dpi=300, bbox_inches='tight')