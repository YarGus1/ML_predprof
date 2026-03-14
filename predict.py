import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
df = pd.read_csv(DATA_URL + "adult.data.csv") #пандас читает и преобразует файл в читаемый формат
df.head()

# 2. Предпросмотр
print(df.head())
print(f"\nРазмер датасета: {df.shape}")

# 3. Целевая переменная
target_column = 'salary'  # ✅ Правильное имя столбца
print(f"\nУникальные значения '{target_column}': {df[target_column].unique()}")

# 4. Кодирование целевой переменной (<=50K -> 0, >50K -> 1)
le = LabelEncoder()
y = le.fit_transform(df[target_column])  # теперь это 0 и 1
print(f"Кодировка: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 5. Признаки (все столбцы кроме target)
X = df.drop(target_column, axis=1)

# 6. Кодирование категориальных признаков (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)
print(f"\nПосле кодирования: {X.shape[1]} признаков")

# 7. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Масштабирование (важно для логистической регрессии)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Обучение моделей классификации
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'Accuracy': acc,
        'model': model,
        'predictions': y_pred,
        'X_test': X_test_scaled if name == 'Logistic Regression' else X_test
    }
    
    print(f"\n📊 {name}:")
    print(f"Точность (Accuracy): {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# 10. Выбор лучшей модели
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = results[best_model_name]['model']
print(f"\n✅ Лучшая модель: {best_model_name} с точностью {results[best_model_name]['Accuracy']:.4f}")

# 11. Визуализация: матрица ошибок
y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Предсказано')
plt.ylabel('Фактически')
plt.title(f'{best_model_name}: Матрица ошибок')

# Визуализация важности признаков (для Random Forest)
if best_model_name == 'Random Forest':
    plt.subplot(1, 2, 2)
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Признак': X.columns,
        'Важность': importances
    }).sort_values('Важность', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='Важность', y='Признак')
    plt.title('Топ-10 важных признаков')

plt.tight_layout()
plt.savefig('priority.png')

# 12. 🔮 Прогноз для новых данных
def predict_salary(new_data_dict, model, model_name, X_columns, scaler, label_encoder):
    """
    Прогнозирование зарплаты для новых данных
    """
    new_df = pd.DataFrame([new_data_dict])
    
    # One-Hot Encoding для новых данных (с теми же колонками)
    new_df = pd.get_dummies(new_df, drop_first=True)
    
    # Добавить отсутствующие колонки со значением 0
    for col in X_columns:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[X_columns]  # Упорядочить колонки как в обучении
    
    # Предсказание
    if model_name == 'Logistic Regression':
        new_scaled = scaler.transform(new_df)
        pred_encoded = model.predict(new_scaled)[0]
    else:
        pred_encoded = model.predict(new_df)[0]
    
    # Декодирование результата
    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    probability = model.predict_proba(new_scaled if model_name == 'Logistic Regression' else new_df)[0]
    
    return prediction, probability

# Пример использования:
# new_person = {
#     'age': 35,
#     'workclass': 'Private',
#     'fnlwgt': 200000,
#     'education': 'Bachelors',
#     'education-num': 13,
#     'marital-status': 'Married-civ-spouse',
#     'occupation': 'Tech-support',
#     'relationship': 'Husband',
#     'race': 'White',
#     'sex': 'Male',
#     'capital-gain': 0,
#     'capital-loss': 0,
#     'hours-per-week': 40,
#     'native-country': 'United-States'
# }
# pred, prob = predict_salary(new_person, best_model, best_model_name, X.columns, scaler, le)
# print(f"\n🔮 Прогноз: {pred}")
# print(f"Вероятности: {dict(zip(le.classes_, prob))}")