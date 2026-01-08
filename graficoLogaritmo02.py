import pandas as pd
import matplotlib.pyplot as plt

# Final results
data = {
    "Model": ["Random Forest", "XGBoost", "Seq2Point"],
    "MAE": [0.013510, 0.932015, 0.624110],
    "RMSE": [0.553513, 13.058208, 1.945092]
}

df = pd.DataFrame(data).set_index("Model")

# Bar plot with logarithmic scale
ax = df.plot(kind="bar")
ax.set_yscale("log")

plt.ylabel("Error (log scale)")
plt.title("Comparison of NILM Models (MAE and RMSE)")
plt.tight_layout()

# Save for LaTeX
plt.savefig("comparison_mae_rmse_log.pdf")
plt.show()
