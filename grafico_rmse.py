import matplotlib.pyplot as plt

models = [
    "Linear Regression", "KNN", "Decision Tree", "Random Forest",
    "Gradient Boosting", "XGBoost", "LightGBM", "MLP", "Seq2Point"
]

rmse = [1.16e-13, 0.4083, 1.0204, 0.5539, 0.3341, 9.6331, 1.8542, 0.0679, 1.9451]
nrmse = [4.01e-17, 0.00014, 0.00035, 0.00019, 0.00012, 0.00333, 0.00064, 0.00002, 0.00067]
r2 = [1.0, 0.999996, 0.999976, 0.999993, 0.999997, 0.997838, 0.999920, 1.0, 0.999912]

def save_plot(values, ylabel, filename):
    plt.figure()
    plt.bar(models, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_plot(rmse, "RMSE", "rmse_comparison.pdf")
save_plot(nrmse, "NRMSE", "nrmse_comparison.pdf")
save_plot(r2, "R²", "r2_comparison.pdf")
