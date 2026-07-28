"""Small numeric, boolean, and perception metric helpers shared by analysis scripts."""

import pandas as pd


DEFAULT_WARNING_DISTANCE_M = 2.0
DEFAULT_DANGER_DISTANCE_M = 1.0

RISK_LEVEL_TO_VALUE = {
    "clear": 0,
    "detected": 1,
    "warning": 2,
    "danger": 3,
}


def first_value(df, column):
    if column not in df.columns:
        return None
    values = df[column].dropna()
    values = values[values.astype(str) != ""]
    if values.empty:
        return None
    return values.iloc[0]


def safe_max(df, column):
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    value = values.max(skipna=True)
    return None if pd.isna(value) else float(value)


def safe_mean(df, column):
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    value = values.mean(skipna=True)
    return None if pd.isna(value) else float(value)


def safe_median(df, column):
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    value = values.median(skipna=True)
    return None if pd.isna(value) else float(value)


def safe_last_valid(df, column):
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def duration_s(df):
    elapsed = df["elapsed_s"].dropna()
    if elapsed.empty:
        return None
    return float(elapsed.max() - elapsed.min())


def format_value(value, unit=""):
    if value is None:
        return "N/A"
    return f"{value:.3f}{unit}"


def ratio(numerator, denominator):
    if denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def bool_from_value(value):
    if value is None:
        return None
    return str(value).strip().lower() in {"true", "1", "yes"}


def has_perception_columns(df):
    if "perception_enabled" not in df.columns:
        return False
    return "perception_risk_level" in df.columns or "detected_obstacle" in df.columns


def perception_enabled_mask(df):
    if "perception_enabled" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["perception_enabled"].map(bool_from_value).fillna(False)


def detected_obstacle_mask(df):
    if "detected_obstacle" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["detected_obstacle"].map(bool_from_value).fillna(False)


def first_active_value(df, column):
    if column not in df.columns:
        return None
    active_df = df[perception_enabled_mask(df)] if "perception_enabled" in df.columns else df
    return first_value(active_df, column)


def risk_level_series(df):
    if "perception_risk_level" in df.columns:
        risk = df["perception_risk_level"].fillna("").astype(str).str.lower()
        if (risk != "").any():
            return risk

    if "detected_obstacle" not in df.columns:
        return pd.Series("", index=df.index)

    detected = detected_obstacle_mask(df)
    warning_distance = first_active_value(df, "warning_distance_m")
    danger_distance = first_active_value(df, "danger_distance_m")
    warning_distance = (
        DEFAULT_WARNING_DISTANCE_M if warning_distance is None else float(warning_distance)
    )
    danger_distance = (
        DEFAULT_DANGER_DISTANCE_M if danger_distance is None else float(danger_distance)
    )

    risk_values = []
    for index, is_detected in detected.items():
        if not is_detected:
            risk_values.append("clear")
            continue
        distance_m = (
            df.loc[index, "nearest_obstacle_distance_m"]
            if "nearest_obstacle_distance_m" in df.columns
            else None
        )
        if pd.isna(distance_m):
            risk_values.append("detected")
        elif float(distance_m) <= danger_distance:
            risk_values.append("danger")
        elif float(distance_m) <= warning_distance:
            risk_values.append("warning")
        else:
            risk_values.append("detected")
    return pd.Series(risk_values, index=df.index)


def frequent_obstacle_names(df, limit=5):
    obstacle_counts = {}
    if "nearest_obstacle_name" not in df.columns:
        return []
    for names_text in df["nearest_obstacle_name"]:
        for name in str(names_text).split(","):
            name = name.strip()
            if not name:
                continue
            obstacle_counts[name] = obstacle_counts.get(name, 0) + 1
    return [
        f"{name} ({count})"
        for name, count in sorted(
            obstacle_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def time_in_risk_levels(df, risk_levels):
    result = {level: 0.0 for level in RISK_LEVEL_TO_VALUE}
    if "elapsed_s" not in df.columns or df.empty:
        return result

    risk_df = pd.DataFrame(
        {
            "elapsed_s": pd.to_numeric(df["elapsed_s"], errors="coerce"),
            "risk_level": risk_levels.reindex(df.index).fillna("").astype(str).str.lower(),
        },
        index=df.index,
    )
    risk_df = risk_df[
        risk_df["elapsed_s"].notna()
        & risk_df["risk_level"].isin(RISK_LEVEL_TO_VALUE)
    ].sort_values("elapsed_s")
    if risk_df.empty:
        return result

    deltas = risk_df["elapsed_s"].shift(-1) - risk_df["elapsed_s"]
    positive_deltas = deltas[deltas > 0]
    median_dt = None if positive_deltas.empty else float(positive_deltas.median())
    max_reasonable_dt = 10.0 if median_dt is None else max(10.0, median_dt * 10.0)

    for risk_level, dt in zip(risk_df["risk_level"], deltas):
        if pd.isna(dt):
            dt = 0.0
        dt = float(dt)
        if dt < 0 or dt > max_reasonable_dt:
            continue
        result[risk_level] += dt
    return result
