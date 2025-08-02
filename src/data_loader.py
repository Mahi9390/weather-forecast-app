import os
import pandas as pd

def load_weather_data():
    """
    Loads the GlobalWeatherRepository.csv file from the data folder.
    Returns a pandas DataFrame.
    """
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the CSV file in ../data/
    csv_path = os.path.join(base_dir, "..", "data", "GlobalWeatherRepository.csv")

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded data from: {csv_path}")
        return df
    except FileNotFoundError:
        print("❌ CSV file not found at:", csv_path)
        return None
    except Exception as e:
        print("❌ Failed to load data:", e)
        return None
