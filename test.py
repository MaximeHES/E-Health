import pandas as pd
import matplotlib.pyplot as plt



# Sample data
data = {
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'New York', 'Chicago'],
    'amount': [250, 150, 200, 300, 100, 400, 150]
}
sales = pd.DataFrame(data)
sales_by_city = sales.groupby('city')['amount'].sum()
sales_by_city.plot(kind='bar')
plt.title('Sales by City')
plt.xlabel('City')
plt.ylabel('Total')
plt.tight_layout()
plt.show()
