import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("TheVanilla\src\Simple_NN\plot_data.csv")
plt.figure(figsize=(10,6))
plt.plot(df['x'], df['true_y'], label='True Values', marker='o')
plt.plot(df['x'], df['predicted_y'], 'rx', label='Predicted Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('True vs Predicted Values')
plt.show()