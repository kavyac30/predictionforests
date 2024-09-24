import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')

# Pie Chart for Target Variable Distribution
# plt.figure(figsize=(10, 7))
# plt.title('Pie Chart: Distribution of Pneumonia Cases')
# df['Fire_Risk'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, wedgeprops=dict(width=0.3))
# plt.ylabel('')  # Remove ylabel for better appearance
# plt.savefig('pie_chart.png')  # Save the pie chart
# plt.show()

# Bar Graph for Average Feature Values
plt.figure(figsize=(10, 7))
plt.title('Bar Chart: Average Feature Values')
feature_means = df[['Temperature', 'Rain', 'Humidity', 'Wind Speed', 'Oxygen']].mean()
feature_means.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.savefig('average_feature_values.png')  # Save the bar chart
plt.show()

# Histogram for Rain Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Rain Distribution')
plt.hist(df['Rain'], bins=20, color='#ff9999', edgecolor='black')
plt.xlabel('Rain')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('rain_distribution_histogram.png')  # Save the histogram
plt.show()

# Histogram for Temperature Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Temperature Distribution')
plt.hist(df['Temperature'], bins=20, color='#66b3ff', edgecolor='black')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('temperature_distribution_histogram.png')  # Save the histogram
plt.show()

# Histogram for Wind Speed Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Wind Speed Distribution')
plt.hist(df['Wind Speed'], bins=20, color='#99ff99', edgecolor='black')
plt.xlabel('Wind Speed')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('wind_speed_distribution_histogram.png')  # Save the histogram
plt.show()

# Histogram for Oxygen Distribution
plt.figure(figsize=(8, 6))
plt.title('Histogram: Oxygen Distribution')
plt.hist(df['Oxygen'], bins=20, color='#ffcc99', edgecolor='black')
plt.xlabel('Oxygen')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('oxygen_distribution_histogram.png')  # Save the histogram
plt.show()

