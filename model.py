import warnings
from datetime import datetime
import pandas as pd
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

START = "2005-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

# Fetch Data from FRED
def get_fred(series_id, start=START, end=TODAY):
    return pdr.DataReader(series_id, "fred", start, end).rename(columns={series_id: series_id})

# CPI (monthly index) --> YoY %
cpi = get_fred("CPIAUCSL")
cpi["CPI_YoY"] = cpi["CPIAUCSL"].pct_change(12) * 100

# Fed Funds (monthly avg %)
fedfunds = get_fred("FEDFUNDS")

# Real GDP (quarterly, annualized) --> YoY % --> forward-fill monthly
gdp = get_fred("GDPC1")
gdp["GDP_YoY"] = gdp["GDPC1"].pct_change(4) * 100
gdp_m = gdp.resample("MS").ffill()[["GDP_YoY"]]

# 10Y Treasury (daily %) --> monthly avg
dgs10 = get_fred("DGS10")
dgs10_m = dgs10.resample("MS").mean().rename(columns={"DGS10": "Yield10Y"})

# Unemployment rate (monthly %)
unrate = get_fred("UNRATE")

# Fed Treasury Holdings (weekly, billions)--> convert to monthly average
# treast=get_fred("TREAST")
# treast_m = treast.resample("MS").mean()

# Combine dataset
df = pd.concat([cpi[["CPI_YoY"]], fedfunds, gdp_m, dgs10_m, unrate], axis=1).dropna()

print("\nSample of dataset:")
print(df.head())

# Features & target
X = df[["CPI_YoY", "FEDFUNDS", "GDP_YoY","UNRATE"]]
y = df["Yield10Y"]

#scaler = StandardScaler() 
#X_scaled = scaler.fit_transform(X)

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Train/test split (80% train, 20% test)
# -----------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X_sm.iloc[:split_idx], X_sm.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train regression on training set
model = sm.OLS(y_train, X_train).fit()

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# MAE
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(model.summary())
print(f"\nTrain MAE: {mae_train:.3f} percentage points")
print(f"Test MAE:  {mae_test:.3f} percentage points")

# Plot Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color="blue", label="Test Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual 10Y Yield (%)")
plt.ylabel("Predicted 10Y Yield (%)")
plt.title("Actual vs Predicted 10Y Treasury Yields (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.index, y, label="Actual Yield", lw=2)
plt.plot(y_train.index, y_pred_train, label="Train Predictions", lw=2, linestyle="--")
plt.plot(y_test.index, y_pred_test, label="Test Predictions", lw=2, linestyle="--")
plt.xlabel("Date")
plt.ylabel("10Y Treasury Yield (%)")
plt.title("Actual vs Predicted 10Y Yields Over Time")
plt.legend()
plt.tight_layout()
plt.show()




