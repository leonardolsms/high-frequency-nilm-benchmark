# ==============================
# NILM Dataset Loader - SBRC
# ==============================

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Redefine regression_metrics to fix the 'squared' argument issue
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Electrical and harmonic features (as in Sensors-25-04601)
ELECTRICAL_COLS = [
    "irms", "vrms", "power_factor",
    "p_apparente", "p_active"
]

HARMONIC_COLS = [f"h{i}" for i in range(1, 33)]

FEATURE_COLS = ELECTRICAL_COLS + HARMONIC_COLS

def load_nilm_dataset(data_root):
    """
    Loads the NILM dataset organized by acquisition sessions and sensor-phase files.

    Dataset structure:
    data/
      ├── session_1/
      │    ├── S1P1.csv
      │    ├── ...
      ├── session_2/
           ├── ...

    Returns
    -------
    pd.DataFrame
        Unified dataframe with metadata columns.
    """
    data_root = Path(data_root)
    dfs = []

    for session_dir in sorted(data_root.iterdir()):
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name

        for csv_file in sorted(session_dir.glob("*.csv")):
            filename = csv_file.stem  # e.g., S1P1
            sensor = filename.split("P")[0]      # S1
            phase = "P" + filename.split("P")[1] # P1

            df = pd.read_csv(csv_file)

            # Keep only relevant columns
            df = df[["time"] + FEATURE_COLS]

            # Metadata (important for analysis)
            df["session"] = session_name
            df["sensor"] = sensor
            df["phase"] = phase
            df["channel"] = f"{sensor}_{phase}"

            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

# If using Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

DATA_PATH = "data"  # ajuste se necessário

df = load_nilm_dataset(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
df.head()
assert all(col in df.columns for col in FEATURE_COLS), "Missing feature columns"
assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

df.describe()
X = df[FEATURE_COLS].values
y = df["p_active"].values  # or appliance label if available

from sklearn.model_selection import train_test_split

train_sessions = df["session"].unique()[:int(0.7 * df["session"].nunique())]

train_idx = df["session"].isin(train_sessions)
test_idx  = ~train_idx

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)



'''def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nrmse = rmse / (y_true.max() - y_true.min())
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, nrmse, r2'''



'''def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # The 'squared' argument was removed from mean_squared_error in scikit-learn 1.0
    # To get RMSE, calculate MSE first and then take its square root.
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_true.max() - y_true.min())
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, nrmse, r2

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

rf_metrics = regression_metrics(y_test, y_pred_rf)



xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

xgb_metrics = regression_metrics(y_test, y_pred_xgb)


def create_seq2point_windows(X, y, window_size=99):
    X_seq, y_seq = [], []
    half = window_size // 2

    for i in range(half, len(X) - half):
        X_seq.append(X[i-half:i+half+1])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 49
BATCH_SIZE = 64
X = X.astype(np.float32)
y = y.astype(np.float32)


X_train_seq, y_train_seq = create_seq2point_windows(X_train_scaled, y_train, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_seq2point_windows(X_test_scaled,  y_test,  WINDOW_SIZE)



class Seq2Point(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 30, kernel_size=10)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=8)
        #self.fc1 = nn.Linear(30 * 82, 1024)
        self.fc1 = nn.Linear(990, 1024)

        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze()

model = Seq2Point(X_train_seq.shape[2])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32)
    ),
    batch_size=64,
    shuffle=True
)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred_seq = model(
        torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    ).cpu().numpy()

seq_metrics = regression_metrics(y_test_seq, y_pred_seq)

results = pd.DataFrame(
    [
        rf_metrics,
        xgb_metrics,
        seq_metrics
    ],
    columns=["MAE", "RMSE", "NRMSE", "R2"],
    index=["Random Forest", "XGBoost", "Seq2Point"]
)

print(results)'''

#results.to_csv("results_comparison.csv")



'''# KNN model
knn = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

# Train
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)

# Metrics
mae_knn = mean_absolute_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
nrmse_knn = rmse_knn / (y_test.max() - y_test.min())
r2_knn = r2_score(y_test, y_pred_knn)

print("KNN Results:")
print("MAE:", mae_knn)
print("RMSE:", rmse_knn)
print("NRMSE:", nrmse_knn)
print("R2:", r2_knn)'''



# SVR model
svr = SVR(
    kernel="rbf",
    C=10,
    epsilon=0.1
)

# Train
svr.fit(X_train, y_train)

# Predict
y_pred_svr = svr.predict(X_test)

# Metrics
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
nrmse_svr = rmse_svr / (y_test.max() - y_test.min())
r2_svr = r2_score(y_test, y_pred_svr)

print("SVR Results:")
print("MAE:", mae_svr)
print("RMSE:", rmse_svr)
print("NRMSE:", nrmse_svr)
print("R2:", r2_svr)




'''# LightGBM model
lgbm = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# Train
lgbm.fit(X_train, y_train)

# Predict
y_pred_lgbm = lgbm.predict(X_test)

# Metrics
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
nrmse_lgbm = rmse_lgbm / (y_test.max() - y_test.min())
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("LightGBM Results:")
print("MAE:", mae_lgbm)
print("RMSE:", rmse_lgbm)
print("NRMSE:", nrmse_lgbm)
print("R2:", r2_lgbm)

#import pandas as pd

results = pd.DataFrame({
    "MAE": {
        "Random Forest": mae_rf,
        "XGBoost": mae_xgb,
        "Seq2Point": mae_seq,
        "KNN": mae_knn,
        "SVR": mae_svr,
        "LightGBM": mae_lgbm
    },
    "RMSE": {
        "Random Forest": rmse_rf,
        "XGBoost": rmse_xgb,
        "Seq2Point": rmse_seq,
        "KNN": rmse_knn,
        "SVR": rmse_svr,
        "LightGBM": rmse_lgbm
    },
    "NRMSE": {
        "Random Forest": nrmse_rf,
        "XGBoost": nrmse_xgb,
        "Seq2Point": nrmse_seq,
        "KNN": nrmse_knn,
        "SVR": nrmse_svr,
        "LightGBM": nrmse_lgbm
    },
    "R2": {
        "Random Forest": r2_rf,
        "XGBoost": r2_xgb,
        "Seq2Point": r2_seq,
        "KNN": r2_knn,
        "SVR": r2_svr,
        "LightGBM": r2_lgbm
    }
})

print(results)

results.to_csv("results_comparison_extended.csv")'''






