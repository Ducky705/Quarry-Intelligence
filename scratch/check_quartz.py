import os
import sys
import pandas as pd

# Path setup
BASE_DIR = "/Users/diegosargent/Documents/Programs/XGBoost-Sniper"
sys.path.append(os.path.join(BASE_DIR, 'src'))

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def check_quartz():
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached()
    
    fe = FeatureEngineer(raw_df)
    df = fe.process()
    
    ms = ModelSimulator(df)
    v4_res = ms.run_v4_quartz()
    
    print(f"Quartz Picks Found: {len(v4_res)}")
    if not v4_res.empty:
        print(v4_res[['pick_date', 'league_name', 'pick_norm', 'profit_actual']].head())
        print(f"Total ROI: {v4_res['profit_actual'].sum() / v4_res['wager_unit'].sum() * 100:.1f}%")
        print(f"Total Bets: {len(v4_res)}")

if __name__ == "__main__":
    check_quartz()
