import os
import sys
import pandas as pd

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def check_sim():
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached()
    fe = FeatureEngineer(raw_df)
    df = fe.process()
    ms = ModelSimulator(df)
    
    quartz = ms.run_v4_quartz()
    print(f"Quartz actual picks: {len(quartz)}")
    if not quartz.empty:
        wins = len(quartz[quartz['outcome'] == 1])
        losses = len(quartz[quartz['outcome'] == 0])
        print(f"Quartz Actual Wins: {wins}")
        print(f"Quartz Actual Losses: {losses}")
        if (wins+losses) > 0:
            print(f"Quartz Actual Win Rate: {wins/(wins+losses)*100:.1f}%")
        print(f"Quartz Actual Net: {quartz['profit_actual'].sum():.2f}")
        print(f"Quartz Actual ROI: {(quartz['profit_actual'].sum()/quartz['wager_unit'].sum()*100):.1f}%")

if __name__ == "__main__":
    check_sim()
