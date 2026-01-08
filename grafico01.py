import pandas as pd
import matplotlib.pyplot as plt

# Final results
data = {
    "Model": ["Random Forest", "XGBoost", "Seq2Point"],
    "MAE": [0.013510, 0.932015, 0.624110],
    "RMSE": [0.553513, 13.058208, 1.945092]
}

df = pd.DataFrame(data).set_index("Model")

# Bar plot
df.plot(kind="bar")
plt.ylabel("Error")
plt.title("Comparison of NILM Models (MAE and RMSE)")
plt.tight_layout()
plt.show()
