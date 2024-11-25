import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
    'temperature': np.random.normal(50, 10, 100),  
    'pressure': np.random.normal(30, 5, 100),      
    'operation_hours': np.arange(100),             
    'failure': np.random.randint(0, 2, 100),       
    'last_maintenance': pd.date_range(start='2022-12-01', periods=100, freq='3D'),
}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['last_maintenance'] = pd.to_datetime(df['last_maintenance'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  
df['is_weekend'] = df['dayofweek'].isin([5, 6])  
df['time_since_last_maintenance'] = (df['timestamp'] - df['last_maintenance']).dt.total_seconds()
window_size = 24
df['rolling_mean_temp'] = df['temperature'].rolling(window=window_size).mean()
df['rolling_std_temp'] = df['temperature'].rolling(window=window_size).std()
df['rolling_max_temp'] = df['temperature'].rolling(window=window_size).max()
df['rolling_min_temp'] = df['temperature'].rolling(window=window_size).min()
df['rolling_mean_pressure'] = df['pressure'].rolling(window=window_size).mean()
df['rolling_std_pressure'] = df['pressure'].rolling(window=window_size).std()
df['lag_1_temp'] = df['temperature'].shift(1)
df['lag_1_pressure'] = df['pressure'].shift(1)
df['lag_1_failure'] = df['failure'].shift(1)
df['lag_2_temp'] = df['temperature'].shift(2)
df['lag_2_pressure'] = df['pressure'].shift(2)
df['lag_2_failure'] = df['failure'].shift(2)
df['cumulative_hours'] = df['operation_hours'].cumsum()
df['cumulative_failures'] = df['failure'].cumsum()  
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)  
df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)  
df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)  
df.fillna(method='bfill', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['rolling_mean_temp'], label='Rolling Mean Temperature (24h)', color='blue')
plt.scatter(df['timestamp'], df['failure']*max(df['rolling_mean_temp']), color='red', label='Failure Events', alpha=0.6)
plt.title('Rolling Mean Temperature and Failure Events')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['time_since_last_maintenance'], label='Time Since Last Maintenance', color='orange')
plt.scatter(df['timestamp'], df['failure']*max(df['time_since_last_maintenance']), color='red', label='Failure Events', alpha=0.6)
plt.title('Time Since Last Maintenance and Failure Events')
plt.xlabel('Timestamp')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
corr = df[['temperature', 'pressure', 'rolling_mean_temp', 'rolling_std_temp', 'time_since_last_maintenance', 'failure']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation with Failure')
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['cumulative_hours'], label='Cumulative Operation Hours', color='green')
plt.plot(df['timestamp'], df['cumulative_failures'], label='Cumulative Failures', color='red')
plt.title('Cumulative Operation Hours vs Cumulative Failures')
plt.xlabel('Timestamp')
plt.ylabel('Cumulative Values')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
print(df.head())
