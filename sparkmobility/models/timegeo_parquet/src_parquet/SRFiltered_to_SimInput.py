import pandas as pd
import numpy as np
from datetime import datetime


# Configure parquet engine to handle PyArrow issues
def configure_parquet_engine():
    """Configure the best available parquet engine."""
    try:
        import pyarrow.parquet

        return "pyarrow"
    except (ImportError, ValueError):
        try:
            import fastparquet

            return "fastparquet"
        except ImportError:
            import os

            os.system("conda install fastparquet -y")
            return "fastparquet"


PARQUET_ENGINE = configure_parquet_engine()


def read_parquet_robust(file_path):
    """Read parquet file with fallback engine support."""
    try:
        return pd.read_parquet(file_path, engine=PARQUET_ENGINE)
    except Exception as e:
        print(f"Error reading {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = "fastparquet" if PARQUET_ENGINE == "pyarrow" else "pyarrow"
        print(f"Trying alternative engine: {alt_engine}")
        return pd.read_parquet(file_path, engine=alt_engine)


def to_parquet_robust(df, file_path, **kwargs):
    """Write parquet file with fallback engine support."""
    try:
        return df.to_parquet(file_path, engine=PARQUET_ENGINE, **kwargs)
    except Exception as e:
        print(f"Error writing {file_path} with {PARQUET_ENGINE}: {e}")
        alt_engine = "fastparquet" if PARQUET_ENGINE == "pyarrow" else "pyarrow"
        print(f"Trying alternative engine: {alt_engine}")
        return df.to_parquet(file_path, engine=alt_engine, **kwargs)


##--------------------------2-Parameters---------------------------
## 3-ParameterValues.py
def decode_and_write_parameters(
    b1_array,
    b2_array,
    commuter_input_path,
    noncommuter_input_path,
    commuter_output_path,
    noncommuter_output_path,
):
    """
    Decode indexed b1 and b2 values in commuter and non-commuter parameter files,
    replacing them with actual numeric values, and write the results to new files.

    Parameters:
        b1_array (list[int]): Mapping array for b1 values.
        b2_array (list[int]): Mapping array for b2 values.
        commuter_input_path (str): Path to the original commuter parameter file.
        noncommuter_input_path (str): Path to the original non-commuter parameter file.
        commuter_output_path (str): Path to the output commuter parameter file.
        noncommuter_output_path (str): Path to the output non-commuter parameter file.
    """

    def process_file(input_path, output_path, b1_array, b2_array):
        with open(output_path, "w") as g:
            with open(input_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 3:
                        continue  # Skip malformed lines
                    try:
                        # Replace index with actual b1 and b2 values
                        parts[0] = str(b1_array[int(parts[0])])
                        parts[1] = str(b2_array[int(parts[1])])
                    except IndexError:
                        print(f"IndexError in line: {line}")
                        continue
                    g.write(" ".join(parts) + "\n")

    # Process both commuter and non-commuter parameter files
    process_file(commuter_input_path, commuter_output_path, b1_array, b2_array)
    process_file(noncommuter_input_path, noncommuter_output_path, b1_array, b2_array)

    print(
        f"Decoded parameters written to:\n  {commuter_output_path}\n  {noncommuter_output_path}"
    )


##--------------------------3_SRFiltered_to_SimInput----------------
## 0_removeRedundance.py


def remove_redundant_stays_parquet(input_path, output_path):
    """
    Remove consecutive stays at the same location from parquet data.

    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to output parquet file
    """
    # Load data
    df = read_parquet_robust(input_path)

    # Sort by user and timestamp
    df = df.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    # Create location identifier combining longitude and latitude
    df["location_id"] = df["Longitude"].astype(str) + "_" + df["Latitude"].astype(str)

    # Remove consecutive stays at same location
    df["prev_user"] = df["caid"].shift(1)
    df["prev_location"] = df["location_id"].shift(1)

    # Keep first record for each user and records where location changed
    mask = (df["caid"] != df["prev_user"]) | (df["location_id"] != df["prev_location"])
    df_filtered = df[mask].copy()

    # Drop helper columns
    df_filtered = df_filtered.drop(
        ["location_id", "prev_user", "prev_location"], axis=1
    )

    # Save as parquet
    to_parquet_robust(df_filtered, output_path, index=False)
    print(f"Removed redundant stays, saved {len(df_filtered)} records to {output_path}")


def parse_parquet_line(row):
    """Parse a parquet row into the format expected by downstream functions"""
    user = str(row["caid"])
    timestamp = int(row["timestamp"])
    trip_purpose = row["type"]
    longitude = str(row["Longitude"])
    latitude = str(row["Latitude"])

    LAtime = datetime.utcfromtimestamp(timestamp)
    date = datetime.strftime(LAtime, "%Y-%m-%d")
    time = float(LAtime.hour) + float(LAtime.minute) / 60 + float(LAtime.second) / 3600

    return [user, date, time, trip_purpose, longitude, latitude]


def extract_frequent_users_parquet(input_path, output_path, num_stays_threshold=15):
    """
    Extract frequent users from parquet data and save as parquet.

    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to output parquet file
        num_stays_threshold (int): Minimum number of distinct stays
    """
    # Load data
    df = read_parquet_robust(input_path)

    # Create location identifier
    df["location_id"] = df["Longitude"].astype(str) + "_" + df["Latitude"].astype(str)

    # Count unique locations per user
    user_location_counts = df.groupby("caid")["location_id"].nunique().reset_index()
    user_location_counts.columns = ["caid", "num_locations"]

    # Filter frequent users
    frequent_users = user_location_counts[
        user_location_counts["num_locations"] > num_stays_threshold
    ]["caid"]

    # Save as parquet
    frequent_users_df = pd.DataFrame({"caid": frequent_users})
    to_parquet_robust(frequent_users_df, output_path, index=False)

    print(f"Saved {len(frequent_users)} frequent users to {output_path}")
    return frequent_users.tolist()


def extract_stay_regions_for_frequent_users_parquet(
    fa_users_path, input_path, output_path
):
    """
    Filter stay regions for frequent users using parquet files.

    Args:
        fa_users_path (str): Path to frequent users parquet file
        input_path (str): Path to input stay regions parquet file
        output_path (str): Path to output parquet file
    """
    # Load frequent users
    frequent_users_df = read_parquet_robust(fa_users_path)
    frequent_users = set(frequent_users_df["caid"].tolist())

    # Load stay regions data
    df = read_parquet_robust(input_path)

    # Filter for frequent users only
    df_filtered = df[df["caid"].isin(frequent_users)].copy()

    # Sort by user and timestamp
    df_filtered = df_filtered.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    # Create location identifier for consecutive duplicate removal
    df_filtered["location_id"] = (
        df_filtered["Longitude"].astype(str) + "_" + df_filtered["Latitude"].astype(str)
    )

    # Remove consecutive duplicates within each user
    df_filtered["prev_user"] = df_filtered["caid"].shift(1)
    df_filtered["prev_location"] = df_filtered["location_id"].shift(1)

    mask = (df_filtered["caid"] != df_filtered["prev_user"]) | (
        df_filtered["location_id"] != df_filtered["prev_location"]
    )
    df_final = df_filtered[mask].copy()

    # Drop helper columns
    df_final = df_final.drop(["location_id", "prev_user", "prev_location"], axis=1)

    # Add derived columns for compatibility
    df_final["date"] = pd.to_datetime(df_final["timestamp"], unit="s").dt.strftime(
        "%Y-%m-%d"
    )
    df_final["time"] = (
        pd.to_datetime(df_final["timestamp"], unit="s").dt.hour
        + pd.to_datetime(df_final["timestamp"], unit="s").dt.minute / 60
        + pd.to_datetime(df_final["timestamp"], unit="s").dt.second / 3600
    )

    # Save as parquet
    to_parquet_robust(df_final, output_path, index=False)
    print(f"Filtered stay regions written to: {output_path}")


def clean_and_format_fa_users_parquet(input_path, output_path):
    """
    Clean and format frequent user data using parquet files.
    """
    # Load data
    df = read_parquet_robust(input_path)

    # Ensure user IDs are always strings
    # df['caid'] = df['caid'].astype(str)

    # Ensure we have required columns
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    if "time" not in df.columns:
        df["time"] = (
            pd.to_datetime(df["timestamp"], unit="s").dt.hour
            + pd.to_datetime(df["timestamp"], unit="s").dt.minute / 60
            + pd.to_datetime(df["timestamp"], unit="s").dt.second / 3600
        )

    # Sort by user and timestamp
    df = df.sort_values(["caid", "timestamp"]).reset_index(drop=True)

    results = []

    for user_id in df["caid"].unique():
        user_df = df[df["caid"] == user_id].copy()
        user_locations = {}
        location_index = 0

        for _, row in user_df.iterrows():
            location_key = (str(row["Longitude"]), str(row["Latitude"]))

            type_val = row["type"]
            if type_val == "home" or type_val == 0 or type_val == "0":
                trip_purpose = "h"
            elif type_val == "work" or type_val == 1 or type_val == "1":
                trip_purpose = "w"
            else:
                trip_purpose = "o"

            # Assign location index
            if location_key not in user_locations:
                if trip_purpose == "h":
                    user_locations[location_key] = "h"
                elif trip_purpose == "w":
                    user_locations[location_key] = "w"
                else:
                    location_index += 1
                    user_locations[location_key] = str(location_index)

            location_id = user_locations[location_key]

            results.append(
                {
                    "caid": user_id,
                    "date": row["date"],
                    "time": row["time"],
                    "trip_purpose": trip_purpose,
                    "Longitude": row["Longitude"],
                    "Latitude": row["Latitude"],
                    "location_index": location_id,
                    "timestamp": row["timestamp"],
                }
            )

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Remove consecutive duplicates
    result_df["prev_user"] = result_df["caid"].shift(1)
    result_df["prev_location"] = result_df["location_index"].shift(1)

    mask = (result_df["caid"] != result_df["prev_user"]) | (
        result_df["location_index"] != result_df["prev_location"]
    )
    result_df = result_df[mask].copy()

    # Drop helper columns
    result_df = result_df.drop(["prev_user", "prev_location"], axis=1)

    # Save as parquet
    to_parquet_robust(result_df, output_path, index=False)
    print(f"Cleaned and formatted data written to: {output_path}")
