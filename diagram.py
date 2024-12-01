import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Set the formula as text
formula = r"$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$"

# Add the text to the plot
ax.text(0.5, 0.5, formula, fontsize=18, ha='center', va='center')

# Remove axes
ax.axis('off')

# Show the plot
plt.title("Accuracy Formula", fontsize=16, fontweight='bold')
plt.show()