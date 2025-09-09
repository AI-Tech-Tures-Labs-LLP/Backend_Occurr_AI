# build_health_js.py
import os
import json
import pandas as pd

# ===================== CONFIG =====================
# Folder that contains Steps.csv, Heart_rate.csv, oxy.csv
FOLDER_PATH = r"C:\Users\phefa\OneDrive\Desktop\Teach Jewellery\Backend_Occurr_AI\fastapp_structure"   # <-- CHANGE THIS

USERNAME = "Antima@123"

FILES = {
    "steps":      "Steps.csv",
    "heartrate":  "Heart_rate.csv",
    "spo2":       "oxy.csv",   # oxy = SpO2
}

# Filter window (inclusive) applied to start_date
START_DATE = "2025-08-01"
END_DATE   = "2025-08-31"
# ==================================================

def csv_path(name: str) -> str:
    return os.path.join(FOLDER_PATH, name)

def main():
    if not os.path.isdir(FOLDER_PATH):
        raise FileNotFoundError(f"Folder not found: {FOLDER_PATH}")

    # --- Make tz-aware UTC bounds for comparison ---
    start_utc = pd.Timestamp(f"{START_DATE} 00:00:00", tz="UTC")
    end_utc   = pd.Timestamp(f"{END_DATE} 23:59:59.999999", tz="UTC")

    all_rows = []

    for metric, filename in FILES.items():
        path = csv_path(filename)
        if not os.path.exists(path):
            print(f"⚠️  Skip '{metric}': file not found → {path}")
            continue

        # Read as raw strings first
        df = pd.read_csv(path)

        needed = {"start_date", "end_date", "value"}
        missing = needed - set(df.columns)
        if missing:
            print(f"❌ '{metric}': Missing columns {missing} in {path}. Found: {list(df.columns)}")
            continue

        # Keep originals for output; create UTC-normalized columns for filtering
        start_raw = pd.to_datetime(df["start_date"], errors="coerce", utc=True)  # aware UTC
        end_raw   = pd.to_datetime(df["end_date"],   errors="coerce", utc=True)  # aware UTC

        # Filter by UTC window on start_date
        mask = (start_raw >= start_utc) & (start_raw <= end_utc)
        filtered_idx = df.index[mask & start_raw.notna()]

        if len(filtered_idx) == 0:
            print(f"ℹ️  {metric}: 0 rows in range")
            continue

        # Build output; parse again WITHOUT forcing UTC to preserve original offset in strings
        for i in filtered_idx:
            ts_local = pd.to_datetime(df.at[i, "start_date"], errors="coerce")
            ca_local = pd.to_datetime(df.at[i, "end_date"],   errors="coerce")

            ts_iso = ts_local.isoformat() if pd.notna(ts_local) else str(df.at[i, "start_date"])
            ca_iso = ca_local.isoformat() if pd.notna(ca_local) else str(df.at[i, "end_date"])

            val = df.at[i, "value"]

            # Convert to plain Python types
            if pd.isna(val):
                val = None
            elif isinstance(val, (pd.Int64Dtype, pd.Float64Dtype)):
                val = val.item()  # pandas nullable scalar to Python type
            elif hasattr(val, "item"):  # NumPy scalar
                val = val.item()
            elif isinstance(val, str):
                try:
                    val = float(val) if "." in val else int(val)
                except Exception:
                    pass

            all_rows.append({
                "metric": metric,
                "timestamp": ts_iso,
                "username": USERNAME,
                "created_at": ca_iso,
                "value": val
            })

        print(f"✅ {metric}: {len(filtered_idx)} rows added from {filename}")

    # Sort by timestamp if possible
    try:
        all_rows.sort(key=lambda x: x["timestamp"])
    except Exception:
        pass

    # Write data.js beside the CSVs
    output_js = os.path.join(FOLDER_PATH, "data.js")
    with open(output_js, "w", encoding="utf-8") as f:
        f.write("const healthData = ")
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
        f.write(";\n\nexport default healthData;\n")

    print(f"📦 Wrote {len(all_rows)} records to {output_js}")

if __name__ == "__main__":
    main()
