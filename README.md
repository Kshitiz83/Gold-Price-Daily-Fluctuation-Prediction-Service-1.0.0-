# 📈 Project 6: Gold Price Daily Fluctuation Prediction Service

An end-to-end production ML system that predicts the daily price fluctuations of gold based on dense macroeconomic indicators, global currencies, index tracking metrics, and chronological trend lag features.

## 🧠 Engineering & Modeling Breakthroughs

### 1. The Core Challenge: Non-Stationarity
* **The Problem**: Initial unregularized Linear/Lasso CV baselines failed drastically on time-series test partitions ($R^2$ of -5.09). This occurred because underlying features (like stock open/close prices) trended continuously upward, causing the linear model to extrapolate values to infinity when handling data outside its training boundaries.
* **The Solution**: Converted the raw, non-stationary target and continuous assets into stationary **Daily First-Differences** ($\Delta x = x_t - x_{t-1}$). This stripped away temporal drift, allowing the model to focus entirely on daily momentum.

### 2. Multi-Stage Multicollinearity Cleansing
* Scaled down an initial overdetermined 80-feature matrix by executing a fast **Upper-Triangle Correlation Filter** (threshold > 0.90) to remove 56 redundant intraday metrics.
* Validated the remaining 24 feature matrices via a statistical **Variance Inflation Factor (VIF)** check to secure stable linear dependencies.

### 3. Tree Ensemble Optimization
* Deployed a constrained **Random Forest Regressor** over the stationary matrices.
* **Final Performance**: Successfully recovered the model out of the negative zone to achieve a stable **Test $R^2$ of 0.5214** and a highly accurate **Test RMSE of $0.5131**. This means the model can predict the daily fluctuation of gold prices to within roughly 51 cents.

### 4. Robust Production API Wrapper
* Built a dynamic FastAPI service featuring defensive input normalization string-mapping.
* The endpoint automatically handles, filters, and orders arbitrary JSON payloads to match Scikit-Learn training schemas flawlessly without throwing shape errors.
