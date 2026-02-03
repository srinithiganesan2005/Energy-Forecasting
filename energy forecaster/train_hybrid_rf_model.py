import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =================================================
# 1. LOAD REAL DATASET (EXCEL)
# =================================================
real_df = pd.read_excel("energy_dataset_1.xlsx")

# Use only required columns
real_df = real_df[[
    "daily_usage_hours",
    "power_rating_watt",
    "daily_energy_kwh"
]]

real_df.columns = ["usage_hours", "power_watt", "energy_kwh"]

print("Real dataset loaded:", real_df.shape)

# =================================================
# 2. GENERATE SIMULATED DATA
# =================================================
np.random.seed(42)
n_simulated = 500

sim_df = pd.DataFrame({
    "usage_hours": np.random.uniform(0.5, 12, n_simulated),
    "power_watt": np.random.uniform(40, 2500, n_simulated)
})

# Physics-based energy formula (realistic)
sim_df["energy_kwh"] = (
    (sim_df["usage_hours"] * sim_df["power_watt"]) / 1000
    + np.random.normal(0, 0.2, n_simulated)
)

print("Simulated dataset generated:", sim_df.shape)

# =================================================
# 3. COMBINE DATASETS
# =================================================
df = pd.concat([real_df, sim_df], ignore_index=True)
print("Combined dataset size:", df.shape)

# =================================================
# 4. FEATURES & TARGET
# =================================================
X = df[["usage_hours", "power_watt"]]
y = df["energy_kwh"]

# =================================================
# 5. TRAIN-TEST SPLIT
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =================================================
# 6. TRAIN HIGH-END RANDOM FOREST MODEL
# =================================================
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# =================================================
# 7. EVALUATION
# =================================================
y_pred = model.predict(X_test)

print("\nHybrid Model Performance")
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))

# =================================================
# 8. FEATURE IMPORTANCE
# =================================================
print("\nFeature Importance")
for f, imp in zip(X.columns, model.feature_importances_):
    print(f"{f}: {round(imp, 3)}")

# =================================================
# 9. SAVE MODEL
# =================================================
joblib.dump(model, "energy_model_hybrid.pkl")
print("\nModel saved as energy_model_hybrid.pkl")
