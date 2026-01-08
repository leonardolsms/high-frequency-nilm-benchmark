import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": [
        "Linear Regression", "KNN", "Decision Tree", "Random Forest",
        "Gradient Boosting", "XGBoost", "LightGBM", "MLP", "Seq2Point"
    ],
    "MAE": [
        5.454709e-14, 4.294270e-02, 5.800851e-02, 1.393337e-02,
        4.912105e-02, 5.797379e-01, 1.768681e-01, 2.974392e-02, 6.241000e-01
    ],
    "RMSE": [
        1.157439e-13, 4.083383e-01, 1.020432e+00, 5.539807e-01,
        3.341337e-01, 9.633088e+00, 1.854185e+00, 6.796672e-02, 1.945100e+00
    ]
}

df = pd.DataFrame(data).set_index("Model")

ax = df.plot(kind="bar", figsize=(10,5))
ax.set_yscale("log")
ax.set_ylabel("Error (log scale)")
ax.set_title("Comparison of NILM Models (MAE and RMSE)")
plt.tight_layout()
plt.savefig("comparison_9models_log.pdf")
plt.show()
