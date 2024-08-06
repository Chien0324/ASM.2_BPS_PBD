import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# a. Data Source.
# Load the CSV files into DataFrames
paths = {
    'customer': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Customer.csv',
    'market_trend': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Market Trend.csv',
    'product_details': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Product Details.csv',
    'product_group': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Product Group.csv',
    'sale': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Sale.csv',
    'website_access_category': 'D:\\.vs\\Assignment\\Business Process Support\\File - Data - BPS\\Website Access Category.csv'
}

data = {name: pd.read_csv(path, on_bad_lines='skip') for name, path in paths.items()}

# b. Data Collection.
# Here data is being collected from CSV files

# c. Data Quality.
# Data Cleaning
customer_df = data['customer']

customer_df['Email'] = customer_df['Email'].str.lower()

customer_df = customer_df[customer_df['Date Of Birth'].str.match(r'\d{2}/\d{2}/\d{4}', na=False)]

customer_df['Date Of Birth'] = pd.to_datetime(customer_df['Date Of Birth'], format='%d/%m/%Y', errors='coerce')

customer_df = customer_df.dropna(subset=['Date Of Birth'])

customer_df['Phone'] = customer_df['Phone'].astype(str)

missing_values_customer = customer_df.isnull().sum()

print("Missing values in customer data after cleaning:")
print(missing_values_customer)

customer_df = customer_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

customer_df.to_csv('customer_cleaned.csv', index=False)

print("Customer data cleaned and saved to 'customer_cleaned.csv'")

# d. Data Storage.
# Data is stored in a cleaned CSV file 'customer_cleaned.csv'

# e. Data Management.
# This part is implicit in the process of cleaning and saving the data, ensuring it's ready for use.

# f. Data Analysis and Processing.
# Sample Data for Visualization
revenue_data = {
    'Month': ['Jun 2023', 'Jul 2023', 'Aug 2023', 'Sep 2023', 'Oct 2023', 'Nov 2023', 'Dec 2023'],
    'Total Price': [25000, 27000, 29000, 30000, 32000, 31000, 15000]
}

products_sold_data = {
    'Month': ['June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Quantity': [50, 55, 53, 52, 57, 55, 25]
}

best_selling_products_data = {
    'Product Name': ['Action Camera', 'Portable Speaker', 'Bluetooth Speaker', 'Electric Toothbrush', 'Wireless Security Camera', 
                     'Smart TV', 'Wireless Mouse'],
    'Sum of Quantity': [15, 14, 13, 12, 11, 10, 9]
}

revenue_analysis_data = {
    'Payment Method': ['Credit Card', 'PayPal', 'Cash'],
    'Amount': [71000, 71000, 54000]
}

# Convert to DataFrames
revenue_df = pd.DataFrame(revenue_data)
products_sold_df = pd.DataFrame(products_sold_data)
best_selling_products_df = pd.DataFrame(best_selling_products_data)
revenue_analysis_df = pd.DataFrame(revenue_analysis_data)

# g. Data Visualization.
# Combined Visualization
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Sales Analysis', fontsize=20)

# Revenue chart over time
sns.lineplot(data=revenue_df, x='Month', y='Total Price', marker='o', ax=axs[0, 0])
axs[0, 0].set_title('Revenue Chart Over Time')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Total Price')
axs[0, 0].grid(True)

# Number of products sold over time
sns.barplot(data=products_sold_df, x='Month', y='Quantity', color='blue', ax=axs[0, 1])
axs[0, 1].set_title('Number of Products Sold Over Time')
axs[0, 1].set_xlabel('Month')
axs[0, 1].set_ylabel('Quantity')

# Best-selling products
sns.barplot(data=best_selling_products_df, x='Sum of Quantity', y='Product Name', palette='rainbow', ax=axs[1, 0])
axs[1, 0].set_title('Best-Selling Products')
axs[1, 0].set_xlabel('Sum of Quantity')
axs[1, 0].set_ylabel('Product Name')

# Revenue analysis by payment method (pie chart)
axs[1, 1].pie(revenue_analysis_df['Amount'], labels=revenue_analysis_df['Payment Method'], autopct='%1.1f%%', colors=['#007bff', 
                                                                                                                      '#6610f2', 
                                                                                                                      '#dc3545'])
axs[1, 1].set_title('Revenue Analysis by Payment Method')

# Adjust layout to prevent overlap
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()

# f. Data Analysis and Processing (Phân Tích và Xử Lý Dữ Liệu) continued
# Sales Prediction
revenue_data = {
    'Month': pd.to_datetime(['2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01']),
    'Total Price': [25000, 27000, 29000, 30000, 32000, 31000, 15000]
}
revenue_df = pd.DataFrame(revenue_data)

# Create additional features
revenue_df['Month_num'] = revenue_df['Month'].dt.month
revenue_df['Year'] = revenue_df['Month'].dt.year

# Prepare the feature matrix X and target vector y
X = revenue_df[['Month_num', 'Year']]
y = revenue_df['Total Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

# Make future predictions
future_months = pd.date_range(start='2024-01-01', end='2024-06-01', freq='MS')
future_df = pd.DataFrame({
    'Month': future_months,
    'Month_num': future_months.month,
    'Year': future_months.year
})

future_predictions = model.predict(future_df[['Month_num', 'Year']])
future_df['Predicted Total Price'] = future_predictions

# g. Data Visualization continued.
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(revenue_df['Month'], revenue_df['Total Price'], label='Historical Sales')
plt.plot(future_df['Month'], future_df['Predicted Total Price'], label='Predicted Sales', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Total Price')
plt.title('Sales Prediction')
plt.legend()
plt.grid(True)
plt.show()

# h. Data Reporting.
# In practice, this step would involve creating detailed reports for stakeholders

# i. Security and Compliance.
# Ensure all data processes comply with relevant security and compliance standards

# j. Conversation and Communication.
# APIs and middleware would facilitate communication between systems and users
