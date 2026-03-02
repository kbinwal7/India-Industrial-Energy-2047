import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
sns.set_theme(style="whitegrid")   
sns.set_palette("Set2")         

def prepare_energy_data(url_df, url_cp_df, url_cf, debug=False):
    # load primary and secondary data
    df = pd.read_csv(url_df)   # primary energy (MT / MMSC)
    cp_df = pd.read_csv(url_cp_df)  # secondary energy (MW)
    
    

    # clean column names
    cp_df.columns = cp_df.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()

    # rename secondary energy columns
    rename_dict = {
        "coal": "coal_e",
        "diesel": "diesel_e",
        "gas": "gas_e",
        "hydro": "hydro_e",
        "solar": "solar_e",
        "wind": "wind_e"
    }
    cp_df.rename(columns=rename_dict, inplace=True)

    # merge datasets
    df = pd.merge(df, cp_df, on="year", how="inner")

    # load conversion factors
    cf_df = pd.read_csv(url_cf)
    cf_df.columns = cf_df.columns.str.strip()
    cf = dict(zip(cf_df["Energy Product (MT)"].str.lower(), cf_df["MTOE/MT"]))
    
    df_orignal=df.copy()

    # apply conversion factors
    for col, factor in cf.items():
        if col in df.columns:
            df[col] = df[col] * factor

    valid_cols = [col for col in cf.keys() if col in df.columns]
    df["fossil_mtoe"] = df[valid_cols].sum(axis=1)

    # secondary energy
    col_e = ['coal_e', 'diesel_e', 'gas_e', 'hydro_e', 'solar_e', 'wind_e']
    for col in col_e:
        if col not in df.columns:
            df[col] = 0

    # CREATE secondary_mtoe directly, no pre-existing reference
    df['non_fossil_mtoe'] = df[col_e].sum(axis=1) * 85.9845*1e-6

    # total energy
    df['total_mtoe'] = df['fossil_mtoe'] + df['non_fossil_mtoe']
    
    #adjust for gdp of industrial sector
    df['gdp(mcr)'] = df['gdp(cr)']*0.27 / 1e6

    # energy intensity
    df['energy_intensity(MJ/kg)'] = (df['total_mtoe'] * 41.87) / (df['production'])
    df['intensity'] = df['total_mtoe'] / df['production']
    
    df['energy_intensity(MJ/cr)']=(df['total_mtoe']*4.187/df['gdp(cr)'])
    df['year_numeric'] = df['year'].astype(str).str.extract(r'(\d{4})').astype(int)

    return df_orignal, df

def trends_energy_consumption(df, x_col="year", y_cols=["primary_mtoe", "secondary_mtoe", "total_mtoe"]):
    def linear(x, a, b): return a + b * x
    def quadratic(x, a, b, c): return a + b*x + c*(x**2)
    def exponential(x, a, b): return a * np.exp(b * x)
    def logarithmic(x, a, b): return a + b * np.log(x + 1)

    models = {
        "Linear": linear,
        "Quadratic": quadratic,
        "Exponential": exponential,
        "Logarithmic": logarithmic
    }

    if df[x_col].dtype == object:
        x_years = df[x_col].str.split("-").str[0].astype(int)
    else:
        x_years = pd.to_numeric(df[x_col], errors="coerce")

    x = x_years - x_years.min()
    results = {}

    fig, axes = plt.subplots(1, len(y_cols), figsize=(6 * len(y_cols), 5))

    for i, col in enumerate(y_cols):
        y = df[col].values
        best_r2, best_model, best_pred, best_params, best_rmse = -np.inf, None, None, None, None

        for name, func in models.items():
            try:
                params, _ = curve_fit(func, x, y, maxfev=10000)
                y_pred = func(x, *params)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                if r2 > best_r2:
                    best_r2, best_model, best_pred, best_params, best_rmse = r2, name, y_pred, params, rmse
            except:
                continue

        results[col] = {
            "Best Model": best_model,
            "R²": best_r2,
            "RMSE": best_rmse,
            "Params": best_params
        }

        ax = axes[i]
        ax.scatter(x_years, y, color="black", s=80, label="Data")
        ax.plot(x_years, best_pred, color="red", lw=2,
                label=f"{best_model}\nR²={best_r2:.3f}, RMSE={best_rmse:.2f}")
        ax.set_title(f"{col}", fontsize=14, weight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return results

def plot_best_trend(df):
    results = trends_energy_consumption(df, x_col="year", y_cols=["primary_mtoe", "secondary_mtoe", "total_mtoe"])
    print("\n=== Best Fit Results ===")
    for col, res in results.items():
        print(f"{col}: {res['Best Model']} (R²={res['R²']:.3f}, RMSE={res['RMSE']:.2f})")

def plot_consumption(df,sector):
  fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

  # Primary Toe area plot
  axes[0].stackplot(df['year'], df['fossil_mtoe'], colors=["tab:blue"], alpha=0.6)
  axes[0].set_title("Fossil fuel consumption in Toe", fontsize=14, weight="bold")
  axes[0].set_xlabel("Year")
  axes[0].set_ylabel("Energy Consumption Toe")
  axes[0].tick_params(axis='x', labelrotation=45)

  # Secondary Toe area plot
  axes[1].stackplot(df['year'], df['non_fossil_mtoe'], colors=["tab:green"], alpha=0.6)
  axes[1].set_title("Non Fossil consumption Toe", fontsize=14, weight="bold")
  axes[1].set_xlabel("Year")
  axes[1].set_ylabel("")
  axes[1].tick_params(axis='x', labelrotation=45)

  # Total Toe stacked area plot (Primary + Secondary)
  axes[2].stackplot(
      df['year'],
      [df['fossil_mtoe'], df['non_fossil_mtoe']],  # wrap in list
      labels=["Fossil Consumption Toe", "Non Fossil Consumption Toe"],
      colors=["tab:blue", "tab:green"],
      alpha=0.6
  )
  axes[2].set_title("Total Consumption", fontsize=14, weight="bold")
  axes[2].set_xlabel("Year")
  axes[2].set_ylabel("Mtoe")  # you may want to add units here
  axes[2].legend(loc='upper left')
  axes[2].tick_params(axis='x', labelrotation=45)

  fig.suptitle("Toe Metrics Area Plots", fontsize=16, weight="bold")
  fig.tight_layout(rect=[0, 0, 1, 0.95])
  fig.suptitle(f"{sector}", fontsize=18, weight="bold")
  plt.show()
  
def plot_all_trends(df, x_col="year", y_cols=["fossil_mtoe", "_mtoe", "total_mtoe"]):
    def linear(x, a, b): return a + b * x
    def quadratic(x, a, b, c): return a + b*x + c*(x**2)
    def exponential(x, a, b): return a * np.exp(b * x)
    def logarithmic(x, a, b): return a + b * np.log(x + 1)

    models = {
        "Linear": linear,
        "Quadratic": quadratic,
        "Exponential": exponential,
        "Logarithmic": logarithmic
    }

    x_parsed = pd.to_numeric(df[x_col].astype(str).str.split("-").str[0], errors="coerce")
    valid_x_mask = ~x_parsed.isna()
    x_years = x_parsed.values.copy()
    x_years = x_years.astype(float)
    x_years[~valid_x_mask] = np.nan

    x_shifted_full = x_years - np.nanmin(x_years)

    n_models = len(models)
    n_y = len(y_cols)
    fig, axes = plt.subplots(n_models, n_y, figsize=(6 * n_y, 4 * n_models), squeeze=False)

    results = {y: {} for y in y_cols}

    for row, (model_name, func) in enumerate(models.items()):
        for col_idx, y_col in enumerate(y_cols):
            y_full = pd.to_numeric(df[y_col], errors="coerce").values
            mask = (~np.isnan(y_full)) & (~np.isnan(x_shifted_full))
            x_fit = x_shifted_full[mask]
            y_fit = y_full[mask]

            ax = axes[row, col_idx]

            if len(x_fit) < 2:
                ax.scatter(x_years, y_full, color="black", s=40, label="Data")
                ax.set_title(f"{model_name} failed for {y_col} (insufficient data)", fontsize=10)
                results[y_col][model_name] = {"R2": None, "RMSE": None, "Params": None, "Success": False}
                ax.grid(alpha=0.3)
                continue

            # initial guesses
            p0 = None
            try:
                if model_name == "Linear":
                    m, c = np.polyfit(x_fit, y_fit, 1)
                    p0 = [c, m]
                elif model_name == "Quadratic":
                    c2, c1, c0 = np.polyfit(x_fit, y_fit, 2)
                    p0 = [c0, c1, c2]
                elif model_name == "Logarithmic":
                    lx = np.log(x_fit + 1)
                    m, c = np.polyfit(lx, y_fit, 1)
                    p0 = [c, m]
                elif model_name == "Exponential":
                    pos = y_fit > 0
                    if pos.sum() >= 2:
                        m, c = np.polyfit(x_fit[pos], np.log(y_fit[pos]), 1)
                        p0 = [np.exp(c), m]
                    else:
                        p0 = [max(y_fit.mean(), 1e-6), 0.0]
            except Exception:
                p0 = np.ones(func.__code__.co_argcount - 1)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    n_params = func.__code__.co_argcount - 1
                    if p0 is None or len(p0) != n_params:
                        p0 = np.ones(n_params)
                    params, _ = curve_fit(func, x_fit, y_fit, p0=p0, maxfev=20000)
                y_pred_fit = func(x_fit, *params)
                y_pred_full = func(x_shifted_full, *params)

                r2 = r2_score(y_fit, y_pred_fit)
                rmse = np.sqrt(mean_squared_error(y_fit, y_pred_fit))

                ax.scatter(x_years, y_full, color="black", s=40, label="Data")
                ax.plot(x_years, y_pred_full, color="red", lw=2, label="Fit")
                ax.set_title(f"{model_name} fit for {y_col}\nR²={r2:.3f}, RMSE={rmse:.2f}", fontsize=10)

                results[y_col][model_name] = {"R2": float(r2), "RMSE": float(rmse), "Params": params.tolist(), "Success": True}
            except Exception as e:
                ax.scatter(x_years, y_full, color="black", s=40, label="Data")
                ax.set_title(f"{model_name} failed for {y_col}", fontsize=10)
                results[y_col][model_name] = {"R2": None, "RMSE": None, "Params": None, "Success": False}

            ax.grid(alpha=0.3)
            if row == n_models - 1:
                ax.set_xlabel("Year")
            if col_idx == 0:
                ax.set_ylabel("Value")

    plt.tight_layout()
    plt.show()
    return results

def plot_all(df):
    return plot_all_trends(df, x_col="year", y_cols=["primary_mtoe", "secondary_mtoe", "total_mtoe"])

def plot_production_gdp(df):
    fig, ax1 = plt.subplots(figsize=(7,5))

    # Bar plot for GDP
    ax1.bar(df["year"], df["gdp(cr)"], color="skyblue", width=0.6,label="GDP (Cr)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("GDP (Cr)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create another axis for Production (line plot)
    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["production"], color="red", marker="o", label="Production")
    ax2.set_ylabel("Production", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title and legends
    plt.title("GDP vs Production over Years")
    fig.tight_layout()
    plt.grid(False)

    plt.show()

def fit_production_gdp(df, future_gdp=None):
    X = np.log(df["gdp(cr)"].values).reshape(-1, 1)
    y = np.log(df["production"].values)

    model = LinearRegression()
    model.fit(X, y)

    beta = model.coef_[0]
    alpha = np.exp(model.intercept_)

    print(f"Fitted model: Production = {alpha:.5f} * GDP^{beta:.5f}")

    # Forecast future production if future GDP values are given
    if future_gdp is not None:
        future_gdp = np.array(future_gdp)
        future_prod = alpha * (future_gdp ** beta)
        forecast_df = pd.DataFrame({
            "GDP(cr)": future_gdp,
            "Forecasted Production": future_prod
        })
        return alpha, beta, forecast_df

    return alpha, beta

def plot_actual_vs_predicted(df, alpha, beta):
    gdp_range = np.linspace(df["gdp(cr)"].min(), df["gdp(cr)"].max() * 1.2, 200)
    predicted = alpha * (gdp_range ** beta)

    plt.figure(figsize=(10, 6))

    # Plot historic data as bars (if you have a year column)
    if "year" in df.columns:
        plt.bar(
            df["gdp(cr)"], 
            df["production"], 
            color="lightgray", 
            alpha=0.6, 
            label="Historic Production"
        )

    # Plot actual data
    plt.scatter(df["gdp(cr)"], df["production"], color="red", s=70, label="Actual Data")

    # Plot predicted curve
    plt.plot(gdp_range, predicted, color="blue", linewidth=2, label="Predicted (Power Law)")

    # Formatting
    plt.xlabel("GDP (Cr)", fontsize=12, fontweight="bold")
    plt.ylabel("Production", fontsize=12, fontweight="bold")
    plt.title("Actual vs Predicted Production from GDP", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
      
def projected_production(alpha, beta, cagr_gdp, gdp, current_year, final_year):
    projected_data = []
    for year in range(final_year - current_year + 1):
        gdp_val = gdp * ((1 + cagr_gdp) ** year)
        production = alpha * (gdp_val ** beta)
        projected_data.append([current_year + year, gdp_val, production])
    return projected_data

def plot_projection(projected_data):
    # Convert list into columns
    years = [row[0] for row in projected_data]
    gdps = [row[1] for row in projected_data]
    productions = [row[2] for row in projected_data]

    fig, ax1 = plt.subplots(figsize=(7,5))

    # Bar plot for GDP
    bars = ax1.bar(years, gdps, color="skyblue", alpha=0.7, label="GDP")
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax1.set_ylabel("GDP", color="blue", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Line plot for Production
    ax2 = ax1.twinx()
    line, = ax2.plot(years, productions, color="red", marker="o", linewidth=2, label="Production")
    ax2.set_ylabel("Production", color="red", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="red")

    # Title and legend
    plt.title("Projected GDP and Production", fontsize=14, fontweight="bold")
    ax1.legend([bars, line], ["GDP", "Production"], loc="upper left", frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
      
def plot_energy_intensity_with_time(df):
    plt.figure(figsize=(8,5))
    plt.plot(df['year'], df['energy_intensity(MJ/kg)'], marker='o', linestyle='-')

    plt.title("Energy Intensity over Time")
    plt.xlabel("Year")
    plt.ylabel("Energy Intensity (e.g., MJ per ton)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()
    
def forecast_cagr(df, feature="total_mtoe", years_ahead=5, method="regression",plot=True):
    
    # --- Clean year column ---
    df = df.copy()
    df["year_num"] = df["year"].str[:4].astype(int)  # take first year in '2016-17'
    df = df.sort_values("year_num")
    consumption = df[feature].reset_index(drop=True)
    years = df["year_num"].values

    start_year, end_year = years[0], years[-1]
    start_value, end_value = consumption.iloc[0], consumption.iloc[-1]
    n_years = end_year - start_year

    # --- CAGR methods ---
    if method == "classic":
        cagr = (end_value / start_value) ** (1 / n_years) - 1

    elif method == "geom_mean":
        growth_rates = consumption.pct_change().dropna() + 1
        cagr = growth_rates.prod() ** (1 / len(growth_rates)) - 1

    elif method == "regression":
        log_values = np.log(consumption.values).reshape(-1, 1)
        X = years.reshape(-1, 1)
        model = LinearRegression().fit(X, log_values)
        beta = model.coef_[0][0]
        cagr = np.exp(beta) - 1
    else:
        raise ValueError("method must be 'classic', 'geom_mean', or 'regression'")

    # --- Forecast ---
    projections = {}
    last_value = end_value
    for i in range(1, years_ahead + 1):
        year = end_year + i
        last_value *= (1 + cagr)
        projections[year] = last_value

    forecast_series = pd.Series(consumption.values, index=years, name="historical")
    projection_series = pd.Series(projections, name="forecast")

    result = pd.concat([forecast_series, projection_series])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(forecast_series.index, forecast_series.values, "o-", label="Historical")
        plt.plot(projection_series.index, projection_series.values, "o--", label="Forecasted")
        plt.xlabel("Year")
        plt.ylabel(feature)
        plt.title(f"{feature} Forecast ({method.capitalize()} CAGR)\nCAGR = {cagr*100:.2f}%")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    return result, cagr
