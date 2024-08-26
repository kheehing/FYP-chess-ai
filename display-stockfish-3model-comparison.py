import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# "Wins": [5, 7, 4, 4, 4, 7, 4, 5, 5],
# "Losses": [4, 3, 4, 5, 3, 3, 5, 5, 3],
# "Draws": [1, 0, 2, 1, 3, 0, 1, 0, 2]

# Example data for table
table_data = {
    "Model": ["Model 1", "Model 1", "Model 1", "Model 2", "Model 2", "Model 2", "Model 3", "Model 3", "Model 3"],
    "ELO": [1000, 1500, 2000, 1000, 1500, 2000, 1000, 1500, 2000],
    "Wins": [41, 42, 46, 49, 36, 52, 49, 46, 52],
    "Losses": [43, 45, 46, 44, 53, 41, 43, 43, 36],
    "Draws": [16, 13, 8, 7, 11, 7, 8, 11, 12]
}



# Create a DataFrame from the data
results_df = pd.DataFrame(table_data)

# Display the table as a heatmap for visualization
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(results_df.pivot_table(index="Model", columns="ELO", values="Wins"), annot=True, fmt=".0f", cmap="Blues", ax=ax)
ax.set_title("Wins by Model Against Stockfish at Different ELO Ratings")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(results_df.pivot_table(index="Model", columns="ELO", values="Losses"), annot=True, fmt=".0f", cmap="Reds", ax=ax)
ax.set_title("Losses by Model Against Stockfish at Different ELO Ratings")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(results_df.pivot_table(index="Model", columns="ELO", values="Draws"), annot=True, fmt=".0f", cmap="Greens", ax=ax)
ax.set_title("Draws by Model Against Stockfish at Different ELO Ratings")
plt.show()
