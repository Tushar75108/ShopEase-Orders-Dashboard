"""
utils.py
--------
Helper functions for the ShopEase Orders Dashboard.

Keeping all the "boring" data-wrangling and stats code in this file (instead of
app.py) keeps the main dashboard file focused on layout/UI. Every function has
a docstring + inline comments explaining WHY a step is done, not just what it
does -- this is meant to be readable by someone learning pandas/Streamlit.
"""

import pandas as pd
import numpy as np

# scipy powers the statistical tests (chi-square, ANOVA/Kruskal-Wallis,
# Shapiro-Wilk). On some locked-down Windows machines, scipy's compiled
# files get blocked by an Application Control / antivirus policy
# (e.g. "DLL load failed while importing _trlib: An Application Control
# policy has blocked this file"). That's a Windows security setting, not a
# bug in this code -- but rather than let it crash the WHOLE dashboard, we
# import scipy defensively: if it fails, SCIPY_AVAILABLE becomes False and
# the stats tab just shows a friendly message instead of the test results.
# Everything else in the app (charts, filters, KPIs, data cleaning) does
# NOT depend on scipy and keeps working either way.
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Same defensive pattern for matplotlib/seaborn (required by the FBDA rubric
# for the Pair Plot and Box-Whisker Plot). These are pure-plotting libraries
# with their own compiled backends, so they can in principle hit the same
# kind of Windows Application Control block scipy did -- guarding the
# import means the rest of the dashboard is never at risk from this.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, required for Streamlit
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# -----------------------------------------------------------------------
# GROUP CONFIG
# -----------------------------------------------------------------------
# Group ID: 108_092_079  ->  random_state is the group id WITHOUT underscores.
# This seed must be used every time we sample, so the same 2,500 rows are
# picked no matter who on the team runs the notebook/app.
GROUP_RANDOM_STATE = 108092079
SAMPLE_SIZE = 2500


# -----------------------------------------------------------------------
# 1. LOAD + SAMPLE
# -----------------------------------------------------------------------
def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Reads the raw ShopEase CSV from disk.

    We keep this separate from cleaning on purpose: if loading fails (wrong
    path, corrupted file, etc.) we want a clear error message instead of a
    confusing crash somewhere deep in the cleaning logic.
    """
    df = pd.read_csv(csv_path)
    return df


def sample_dataset(df: pd.DataFrame, n: int = SAMPLE_SIZE,
                    random_state: int = GROUP_RANDOM_STATE) -> pd.DataFrame:
    """
    Randomly samples n rows using our group's fixed seed.

    Using a fixed random_state means every group member (and the grader,
    if they re-run the notebook) gets the EXACT same 2,500 rows every time.
    Without a fixed seed, pandas would pick a different random sample on
    every run, and our numbers would never match across teammates.
    """
    # If the file somehow has fewer rows than we want to sample, just
    # return everything instead of crashing.
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


# -----------------------------------------------------------------------
# 2. CLEANING
# -----------------------------------------------------------------------
def _normalize_text(series: pd.Series) -> pd.Series:
    """
    Strips whitespace and standardizes casing for a text column.
    e.g. " FEMALE ", "female", "Female" all become "Female".
    """
    return series.astype(str).str.strip().str.title()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline for the ShopEase orders sample.

    Returns:
        cleaned_df: the cleaned dataframe, ready for analysis/plotting
        quality_report: a dict of counts describing what was fixed/dropped,
                         used later in the "Data Quality & Tests" tab so we
                         can SHOW the grader what we cleaned instead of just
                         claiming it in the report.
    """
    df = df.copy()
    quality_report = {}

    # --- Text column normalization (fixes "Cairo" vs "CAIRO" vs " cairo ") ---
    text_cols = ["Gender", "City", "Category", "PaymentMethod", "OrderStatus"]
    for col in text_cols:
        df[col] = _normalize_text(df[col])

    # Gender has extra short forms ("F", "M") on top of casing issues.
    df["Gender"] = df["Gender"].replace({"F": "Female", "M": "Male"})

    # "Canceled" (one L) vs "Cancelled" (two L) are the same status.
    df["OrderStatus"] = df["OrderStatus"].replace({"Canceled": "Cancelled"})

    # --- OrderDate: some rows literally contain the string "not available" ---
    # errors="coerce" turns anything unparseable (including "not available")
    # into NaT (pandas' "missing datetime") instead of crashing the whole load.
    before = df["OrderDate"].isna().sum()
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
    quality_report["order_date_unparseable"] = int(df["OrderDate"].isna().sum() - before)

    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")

    # --- Discount: mixed formats, e.g. "0.1" (already a fraction) vs
    # "10%" (a percent string). We normalize everything to a 0-1 float. ---
    def _parse_discount(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val.endswith("%"):
            # "10%" -> 10 -> 0.10
            return float(val.replace("%", "")) / 100.0
        return float(val)

    df["Discount"] = df["Discount"].apply(_parse_discount)

    # --- Drop rows with impossible / invalid values ---
    # We track how many rows each rule removes so we can report it honestly
    # rather than silently deleting data.
    n_start = len(df)

    invalid_mask = (
        (df["Quantity"] <= 0) |
        (df["UnitPrice"] < 0) |
        (df["CustomerAge"] < 0) | (df["CustomerAge"] > 100) |
        (df["Rating"].notna() & ~df["Rating"].between(1, 5)) |
        (df["OrderDate"].isna())
    )
    quality_report["rows_flagged_invalid"] = int(invalid_mask.sum())
    quality_report["rows_before_cleaning"] = n_start

    df_clean = df[~invalid_mask].copy()
    quality_report["rows_after_cleaning"] = len(df_clean)

    # --- Fill remaining "soft" missing values ---
    # CustomerAge: a handful of rows are missing (not invalid, just blank).
    # Median is a safer fill than mean here since age can be skewed by
    # a few very old/young customers.
    df_clean["CustomerAge"] = df_clean["CustomerAge"].fillna(df_clean["CustomerAge"].median())

    # Rating and DeliveryDate missingness is EXPECTED (order not yet
    # delivered / no review left) -- we intentionally do NOT fill these,
    # since doing so would invent data that doesn't exist.

    # CustomerID / Product / PaymentMethod: small number missing, label clearly
    # instead of dropping more rows than necessary.
    df_clean["CustomerID"] = df_clean["CustomerID"].fillna("Unknown")
    df_clean["Product"] = df_clean["Product"].fillna("Unknown")
    df_clean["PaymentMethod"] = df_clean["PaymentMethod"].fillna("Unknown")

    # --- Feature engineering ---
    df_clean["OrderMonth"] = df_clean["OrderDate"].dt.to_period("M").astype(str)
    df_clean["OrderWeekday"] = df_clean["OrderDate"].dt.day_name()

    # Sanity-check TotalAmount against Quantity * UnitPrice * (1 - Discount).
    # We don't overwrite TotalAmount (it's given data), just flag big mismatches
    # for the "Data Quality" tab.
    expected_total = df_clean["Quantity"] * df_clean["UnitPrice"] * (1 - df_clean["Discount"].fillna(0))
    mismatch = (df_clean["TotalAmount"] - expected_total).abs() > 1.0
    quality_report["total_amount_mismatches"] = int(mismatch.sum())

    return df_clean, quality_report


# -----------------------------------------------------------------------
# 3. KPI HELPERS
# -----------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """Returns the headline numbers shown in the top st.metric row."""
    if df.empty:
        return {"total_orders": 0, "total_revenue": 0, "avg_order_value": 0, "top_category": "N/A"}

    return {
        "total_orders": len(df),
        "total_revenue": df["TotalAmount"].sum(),
        "avg_order_value": df["TotalAmount"].mean(),
        "top_category": df["Category"].value_counts().idxmax(),
    }


# -----------------------------------------------------------------------
# 4. STATISTICAL TESTS
# -----------------------------------------------------------------------
def chi_square_independence(df: pd.DataFrame, col1: str, col2: str):
    """
    Chi-square test of independence between two categorical columns.
    Tests: "Is PaymentMethod related to OrderStatus, or are they independent?"
    Returns (chi2_stat, p_value, contingency_table). If scipy isn't
    available, chi2_stat and p_value come back as None -- the contingency
    table (built with plain pandas) still works either way.
    """
    contingency = pd.crosstab(df[col1], df[col2])
    if not SCIPY_AVAILABLE:
        return None, None, contingency
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p, contingency


def anova_or_kruskal(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Tests whether `value_col` differs significantly across groups in `group_col`.
    We run Shapiro-Wilk first to decide: normal data -> ANOVA (parametric),
    non-normal data -> Kruskal-Wallis (non-parametric, safer default for
    messy real-world data like ours).
    Returns (test_name, stat, p_value). If scipy isn't available, returns
    ("unavailable", None, None).
    """
    if not SCIPY_AVAILABLE:
        return "unavailable", None, None

    groups = [g.dropna().values for _, g in df.groupby(group_col)[value_col]]
    groups = [g for g in groups if len(g) > 3]  # need enough points per group

    # Quick normality check on the overall column (Shapiro-Wilk works best
    # on samples < 5000; we subsample if needed).
    sample = df[value_col].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=GROUP_RANDOM_STATE)
    _, normal_p = stats.shapiro(sample)

    if normal_p > 0.05:
        stat, p = stats.f_oneway(*groups)
        return "ANOVA", stat, p
    else:
        stat, p = stats.kruskal(*groups)
        return "Kruskal-Wallis", stat, p


def normality_test(series: pd.Series):
    """
    Shapiro-Wilk test of normality on a numeric column (e.g. Rating).
    Returns (stat, p_value). p < 0.05 means "likely NOT normally distributed".
    Returns (None, None) if scipy isn't available.
    """
    if not SCIPY_AVAILABLE:
        return None, None

    sample = series.dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=GROUP_RANDOM_STATE)
    stat, p = stats.shapiro(sample)
    return stat, p


def correlation_summary(df: pd.DataFrame, cols: list):
    """Returns Pearson and Spearman correlation matrices for the given numeric columns."""
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    return pearson, spearman


# -----------------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------
def descriptive_stats_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Full descriptive-statistics table for numeric (non-categorical) columns:
    count, min, max, mean, median, mode, std dev, skewness, kurtosis, and the
    25th/75th percentiles -- exactly the "Non-Categorical Data" row of the
    FBDA rubric's Analysis-of-Data table.

    Note: .skew() and .kurtosis() are pandas' own implementations (not scipy),
    so this works even on machines where scipy is blocked.
    """
    rows = []
    for col in cols:
        s = df[col].dropna()
        mode_val = s.mode()
        rows.append({
            "Column": col,
            "Count": s.count(),
            "Min": s.min(),
            "Max": s.max(),
            "Mean": s.mean(),
            "Median": s.median(),
            "Mode": mode_val.iloc[0] if not mode_val.empty else None,
            "Std Dev": s.std(),
            "Skewness": s.skew(),
            "Kurtosis": s.kurtosis(),
            "25th %ile": s.quantile(0.25),
            "75th %ile": s.quantile(0.75),
        })
    return pd.DataFrame(rows).set_index("Column")


def descriptive_stats_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Descriptive statistics for categorical columns: count, number of unique
    categories, and the category with the highest/lowest frequency (with
    relative frequency %) -- the "Categorical Data" row of the rubric table.
    """
    rows = []
    for col in cols:
        counts = df[col].value_counts()
        rel = df[col].value_counts(normalize=True)
        rows.append({
            "Column": col,
            "Count": df[col].count(),
            "Unique Categories": df[col].nunique(),
            "Highest-Freq Category": counts.idxmax(),
            "Highest Freq (count)": counts.max(),
            "Highest Freq (%)": round(rel.max() * 100, 1),
            "Lowest-Freq Category": counts.idxmin(),
            "Lowest Freq (count)": counts.min(),
        })
    return pd.DataFrame(rows).set_index("Column")


# -----------------------------------------------------------------------
# 6. REGRESSION (causal analysis)
# -----------------------------------------------------------------------
def linear_regression_ols(df: pd.DataFrame, target_col: str, feature_cols: list) -> dict:
    """
    Simple multiple linear regression (Ordinary Least Squares), implemented
    with plain numpy -- deliberately NOT using scipy or statsmodels, so this
    keeps working even on a machine where scipy is blocked (e.g. the
    Windows Application Control / _trlib DLL issue).

    How it works: we solve for the coefficients that minimize squared error
    using numpy.linalg.lstsq, which is the same underlying math regression
    libraries use, just without the extra dependency.

    Returns a dict with: intercept, coefficients (per feature), r_squared,
    and n_obs (number of rows used, after dropping missing values).
    """
    data = df[[target_col] + feature_cols].dropna()
    X = data[feature_cols].to_numpy(dtype=float)
    y = data[target_col].to_numpy(dtype=float)
    n = X.shape[0]

    # Add a column of 1s so the model can fit an intercept term.
    X_design = np.column_stack([np.ones(n), X])

    # Least-squares solve: finds the coefficients minimizing sum of squared errors.
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    y_pred = X_design @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {
        "intercept": beta[0],
        "coefficients": dict(zip(feature_cols, beta[1:])),
        "r_squared": r_squared,
        "n_obs": n,
    }


# -----------------------------------------------------------------------
# 7. INTERACTIVE EXPLORATION HELPERS (Relationships tab + Regression explorer)
# -----------------------------------------------------------------------
# The numeric columns a user is allowed to pick from dropdowns. Kept as one
# list so the Relationships tab and the Regression explorer always offer
# the same choices.
NUMERIC_COLUMNS = ["CustomerAge", "Quantity", "UnitPrice", "Discount", "TotalAmount", "Rating"]


def simple_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """
    Simple (one predictor) linear regression: y = intercept + slope * x.
    Built with plain numpy -- same reasoning as linear_regression_ols above,
    this keeps working even if scipy is blocked on this machine.

    The only piece that genuinely needs scipy is the p-value for the slope
    (it requires the Student's t-distribution's CDF). If scipy isn't
    available, we still return the slope/intercept/R² and simply set
    p_value to None -- the caller can show "unavailable" for just that one
    number instead of losing the whole regression.

    Returns a dict with: slope, intercept, r_squared, se_slope, t_stat,
    p_value (or None), n (rows used after dropping missing values).
    """
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    n = len(x)

    if n < 3:
        return {"slope": None, "intercept": None, "r_squared": None,
                "se_slope": None, "t_stat": None, "p_value": None, "n": n}

    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    slope = ss_xy / ss_xx if ss_xx != 0 else float("nan")
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    # Standard error of the slope, and its t-statistic -- standard formulas,
    # no scipy needed for these two.
    dof = n - 2
    mse = ss_res / dof if dof > 0 else float("nan")
    se_slope = (mse / ss_xx) ** 0.5 if ss_xx != 0 and dof > 0 else float("nan")
    t_stat = slope / se_slope if se_slope not in (0, float("nan")) else float("nan")

    # The p-value needs the t-distribution's CDF, which is the one part
    # that does need scipy.
    p_value = None
    if SCIPY_AVAILABLE and dof > 0 and se_slope > 0:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    return {
        "slope": slope, "intercept": intercept, "r_squared": r_squared,
        "se_slope": se_slope, "t_stat": t_stat, "p_value": p_value, "n": n,
    }


# -----------------------------------------------------------------------
# 8. MATPLOTLIB / SEABORN VISUALS (Pair Plot, Box-Whisker Plot)
# -----------------------------------------------------------------------
# The FBDA rubric explicitly names Matplotlib and Seaborn as required
# libraries, and separately lists "Pair Plot" and "Box-Whisker Plot" as
# required visualizations. These two functions cover both requirements
# at once. Each returns None if matplotlib/seaborn failed to import, so
# the calling code can show a friendly message instead of crashing.
def pairplot_figure(df: pd.DataFrame, cols: list, hue_col: str = "Category"):
    """
    Seaborn pair plot: every numeric column plotted against every other,
    colored by category. Useful for spotting relationships across several
    variables at once rather than one pair at a time.
    """
    if not MPL_AVAILABLE:
        return None
    n_hues = df[hue_col].nunique()
    grid = sns.pairplot(df[cols + [hue_col]], hue=hue_col,
                         palette=CATEGORY_COLOR_LIST[:n_hues], diag_kind="hist")
    return grid.fig


def boxwhisker_figure(df: pd.DataFrame, value_col: str, group_col: str):
    """Seaborn box-and-whisker plot of value_col split by group_col."""
    if not MPL_AVAILABLE:
        return None
    n_groups = df[group_col].nunique()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=group_col, y=value_col, hue=group_col,
                palette=CATEGORY_COLOR_LIST[:n_groups], legend=False, ax=ax)
    ax.set_title(f"{value_col} by {group_col}")
    plt.xticks(rotation=20)
    return fig


# Shared color list so matplotlib/seaborn charts use roughly the same
# palette as the Plotly charts elsewhere in the app (kept as a plain list
# since seaborn's `palette` argument takes a sequence, not a name->color map).
CATEGORY_COLOR_LIST = ["#4C6EF5", "#22B8CF", "#40C057", "#F76707", "#E64980", "#845EF7"]


# -----------------------------------------------------------------------
# 9. CONFIDENCE INTERVAL
# -----------------------------------------------------------------------
def confidence_interval_mean(series: pd.Series, confidence: float = 0.95) -> dict:
    """
    Confidence interval for a column's mean.
    Uses the t-distribution's critical value when scipy is available (the
    statistically correct approach for smaller samples); falls back to the
    standard normal approximation (z = 1.96 for 95%) when scipy is blocked,
    which is very close to the t-value for our sample sizes (n well over 30).
    """
    sample = series.dropna()
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / (n ** 0.5)

    if SCIPY_AVAILABLE:
        crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        method = "t-distribution"
    else:
        crit = 1.96 if confidence == 0.95 else 2.576  # common z critical values
        method = "normal approximation (z)"

    margin = crit * se
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin,
            "margin": margin, "n": n, "method": method, "confidence": confidence}


# -----------------------------------------------------------------------
# 10. TEST OF VARIANCE (Levene's test)
# -----------------------------------------------------------------------
def levene_test(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Levene's test: are the variances of value_col equal across the groups
    in group_col? (e.g. "is price equally spread out across categories, or
    is one category much more variable than the others?")
    Returns (stat, p_value), or (None, None) if scipy is unavailable.
    """
    if not SCIPY_AVAILABLE:
        return None, None
    groups = [g.dropna().values for _, g in df.groupby(group_col)[value_col]]
    groups = [g for g in groups if len(g) > 1]
    stat, p = stats.levene(*groups)
    return stat, p


# -----------------------------------------------------------------------
# 11. TEST OF CORRELATION (significance of r)
# -----------------------------------------------------------------------
def correlation_significance(df: pd.DataFrame, col1: str, col2: str, method: str = "pearson"):
    """
    Tests whether a correlation coefficient is significantly different from
    zero (i.e. is the relationship real, or could it be due to chance?).
    Returns (r, p_value), or (None, None) if scipy is unavailable.
    """
    if not SCIPY_AVAILABLE:
        return None, None
    data = df[[col1, col2]].dropna()
    if method == "pearson":
        r, p = stats.pearsonr(data[col1], data[col2])
    else:
        r, p = stats.spearmanr(data[col1], data[col2])
    return r, p


# -----------------------------------------------------------------------
# 12. NON-PARAMETRIC: MANN-WHITNEY U TEST
# -----------------------------------------------------------------------
def mann_whitney_test(df: pd.DataFrame, group_col: str, value_col: str, group_a: str, group_b: str):
    """
    Mann-Whitney U test: compares value_col between exactly two groups
    without assuming normal distributions (e.g. "is TotalAmount different
    between Male and Female customers?").
    Returns (stat, p_value), or (None, None) if scipy is unavailable.
    """
    if not SCIPY_AVAILABLE:
        return None, None
    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    stat, p = stats.mannwhitneyu(a, b)
    return stat, p


# -----------------------------------------------------------------------
# 13. POLYNOMIAL REGRESSION
# -----------------------------------------------------------------------
def polynomial_regression(df: pd.DataFrame, x_col: str, y_col: str, degree: int = 2) -> dict:
    """
    Polynomial regression (y as a function of x, x^2, ..., x^degree) using
    numpy.polyfit -- no scipy/sklearn needed. Returns the fitted
    coefficients (highest power first, matching numpy's convention) and R².
    """
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {"coefficients": coeffs, "degree": degree, "r_squared": r_squared, "n": len(x)}


# -----------------------------------------------------------------------
# 14. REGRESSION WITH CATEGORICAL VARIABLES (dummy encoding)
# -----------------------------------------------------------------------
def regression_with_categorical(df: pd.DataFrame, target_col: str, numeric_features: list, cat_col: str) -> dict:
    """
    Multiple regression that includes a categorical predictor by one-hot
    encoding it (dropping the first category as the baseline, standard
    practice to avoid redundant columns). E.g. TotalAmount explained by
    UnitPrice/Quantity AND which Category the order belongs to.

    Built with plain numpy (via linear_regression_ols-style least squares),
    same scipy-independent approach as the rest of this file.
    """
    data = df[[target_col] + numeric_features + [cat_col]].dropna()
    dummies = pd.get_dummies(data[cat_col], prefix=cat_col, drop_first=True, dtype=float)
    dummy_cols = list(dummies.columns)

    X = pd.concat([data[numeric_features].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    y = data[target_col].to_numpy(dtype=float)
    n = len(y)

    X_design = np.column_stack([np.ones(n), X.to_numpy(dtype=float)])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    y_pred = X_design @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    all_feature_names = numeric_features + dummy_cols
    return {
        "intercept": beta[0],
        "coefficients": dict(zip(all_feature_names, beta[1:])),
        "baseline_category": sorted(data[cat_col].unique())[0],
        "r_squared": r_squared,
        "n_obs": n,
    }


# -----------------------------------------------------------------------
# 15. LOGISTIC REGRESSION (binary classification)
# -----------------------------------------------------------------------
def logistic_regression_manual(df: pd.DataFrame, target_binary_col: str, feature_cols: list,
                                learning_rate: float = 0.1, n_iter: int = 3000) -> dict:
    """
    Logistic regression via gradient descent, implemented with plain numpy
    only (no scipy/sklearn) -- consistent with the rest of this file's
    scipy-independent approach, and avoids adding a new dependency that
    could hit the same Windows DLL-blocking issue.

    target_binary_col must already be 0/1 (e.g. IsCancelled). Features are
    standardized (zero mean, unit variance) before training, which is
    standard practice for gradient descent -- it converges far more
    reliably than on raw, differently-scaled columns.

    Returns coefficients (on the standardized scale), intercept, training
    accuracy, and the feature means/stds (needed to interpret or reuse
    the coefficients).
    """
    data = df[[target_binary_col] + feature_cols].dropna()
    X_raw = data[feature_cols].to_numpy(dtype=float)
    y = data[target_binary_col].to_numpy(dtype=float)
    n, n_features = X_raw.shape

    # Standardize features so gradient descent converges well.
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    stds[stds == 0] = 1  # avoid divide-by-zero for a constant column
    X = (X_raw - means) / stds

    weights = np.zeros(n_features)
    bias = 0.0

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    for _ in range(n_iter):
        z = X @ weights + bias
        preds = sigmoid(z)
        error = preds - y
        grad_w = (X.T @ error) / n
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    final_preds = sigmoid(X @ weights + bias)
    accuracy = ((final_preds >= 0.5).astype(int) == y).mean()

    return {
        "weights": dict(zip(feature_cols, weights)),
        "bias": bias,
        "accuracy": accuracy,
        "n": n,
        "feature_means": dict(zip(feature_cols, means)),
        "feature_stds": dict(zip(feature_cols, stds)),
    }
