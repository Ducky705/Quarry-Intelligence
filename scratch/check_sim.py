
import sys
import os
import pandas as pd
import numpy as np

# Path setup
BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, 'src'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def check_pyrite():
    print("Checking Pyrite Simulation...")
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached()
    
    fe = FeatureEngineer(raw_df)
    df = fe.process()
    
    ms = ModelSimulator(df)
    pyrite_res = ms.run_v1_pyrite()
    
    if pyrite_res.empty:
        print("Pyrite results empty!")
        return
        
    net_profit = pyrite_res['profit_actual'].sum()
    roi = (net_profit / pyrite_res['wager_unit'].sum() * 100) if pyrite_res['wager_unit'].sum() > 0 else 0
    
    print(f"Pyrite Actual Stats:")
    print(f"  Net Profit: {net_profit:.2f}u")
    print(f"  ROI: {roi:.2f}%")
    print(f"  Sample: {len(pyrite_res)}")
    
    # Check if it's trending down
    pyrite_res = pyrite_res.sort_values('pick_date')
    pyrite_res['cum_profit'] = pyrite_res['profit_actual'].cumsum()
    
    first_profit = pyrite_res['cum_profit'].iloc[0]
    last_profit = pyrite_res['cum_profit'].iloc[-1]
    
    print(f"  Trend: {first_profit:.2f} -> {last_profit:.2f}")
    if last_profit < 0:
        print("  ❌ TREND IS NEGATIVE")
    else:
        print("  ✅ TREND IS POSITIVE")

if __name__ == "__main__":
    check_pyrite()
