import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, shapiro

data =  {
    "product": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch"],
    "price": [1200, 800, 400, 150, 250]
}

df = pd.DataFrame(data)

mean_price = df["price"].mean()
print(f"Mean price: {mean_price}")

median_price = df["price"].median()
print(f"Median price: {median_price}")

mode_price = df["price"].mode()[0]
print(f"Mode price: {mode_price}")

sns.histplot(df["price"], bins=5, kde=True)
plt.title("Price Distribution of Products")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()  


# test null hypothesis
t_stat, p_value = ttest_1samp(df["price"], 500)
print(f"t-statistic: {t_stat}, p-value: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis: The mean price is significantly different from 500.")
else:
    print("Fail to reject the null hypothesis: The mean price is not significantly different from 500.")
    
    
# test distribution of prices
stat, p = shapiro(df["price"])
print(f"Shapiro-Wilk test statistic: {stat}, p-value: {p}")

if p > 0.05:
    print("The price data is normally distributed.")    
else:
    print("The price data is not normally distributed.")