import pandas as pd
from sklearn.model_selection import train_test_split

# data = pd.read_csv('real-estate_prices.csv')
data = pd.read_csv(r'C:\Users\user\Documents\skill-up\07-hybrid\real-estate_prices.csv')
print(f"{len(data)} строк")
print(data.head())

X = data[['area', 'bedrooms', 'age']] # признаки (входные данные, на основе которых модель делает прогнозы)
y = data['price'] # целевая переменная - значение, которое модель пытается предсказать

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# регрессия

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()

model.fit(X_train, y_train) # обучение модели на тренировочных данных

model.predict(X_test) # прогнозирование цен на тестовых данных

mse = mean_squared_error(y_test, model.predict(X_test)) # вычисление средней квадратичной ошибки
r2 = r2_score(y_test, model.predict(X_test)) # вычисление коэффициента детерминации R^2

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
for feature, coef in zip(X.columns, model.coef_):
    print(f"Feature: {feature}, Coefficient: {coef}")

print("Intercept:", model.intercept_) # свободный член


# классификация

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = {
    'area': [1500, 2200, 800, 1100, 2500],
    'bedrooms': [1, 2, 2, 3, 4],
    'age': [5, 10, 15, 20, 25],
    'category': ['Standard', 'Luxury', 'Affordable', 'Affordable', 'Luxury']
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'age']] # признаки
y = df['category'] # целевая переменная 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train) # обучение модели на тренировочных данных
y_pred = model.predict(X_test) # прогнозирование категорий на тестовых данных

print("Predicted categories:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()


# кластеризация

from sklearn.cluster import KMeans
import seaborn as sns

data2 = pd.read_csv(r'C:\Users\user\Documents\skill-up\07-hybrid\real-estate_prices.csv')
# data = pd.read_csv('real-estate_prices.csv')
print(f"{len(data2)} строк")
print(data2.head())

X = data2[['area', 'bedrooms', 'age']] # признаки

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42) 

# для определения оптимального количества кластеров используем метод локтя (Elbow Method)
# он позволяет увидеть после какого значения количества кластеров улучшение качества кластеризации становится незначительным
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
data2['cluster'] = kmeans.fit_predict(X) # обучение модели и присвоение кластеров каждому объекту
kmeans.fit(X_train) # обучение модели на тренировочных данных

plt.figure(figsize=(8, 5))
sns.scatterplot(data=data2, x='area', y='price', hue='cluster', palette='Set1')
plt.title('KMeans Clustering of Real Estate Prices')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()