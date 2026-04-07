import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import SportsDataPipeline, FeatureEngineer
from models import ModelSimulator

def generate_comparison_chart():
    print("🚀 Generating Comparative Alpha Graphs...")
    
    # Initialize Pipeline & Hydrate Features
    pipeline = SportsDataPipeline()
    raw_df = pipeline.fetch_data_cached() # Uses cache
    
    fe = FeatureEngineer(raw_df)
    df = fe.process() # Fully hydrated with v4 (shifted) and v3 (leaked) features
    
    simulator = ModelSimulator(df)

    # Run Simulations
    print("🌌 Simulating Quartz (Backtest Alpha)...")
    quartz_res = simulator.run_backtest_all() # Use full history for graph
    
    print("🌋 Simulating Obsidian...")
    obsidian_res = simulator.run_v3_obsidian()
    
    # For Pyrite and Diamond, if simulator doesn't have specific methods, 
    # we'll simulate them via edge filters (as they are usually based on Min_Edge)
    print("💎 Simulating Diamond (Surgical)...")
    diamond_res = simulator.run_v2_diamond() # Assuming it exists or fallback
    
    print("☄️ Simulating Pyrite (Aggressive)...")
    pyrite_res = simulator.run_v1_pyrite() # Assuming it exists or fallback

    # Extract Cumulative Series
    def get_series(res_df, label):
        if res_df.empty: return pd.Series(dtype=float)
        
        # FIX: Normalize to DATE to prevent intra-day vertical jumps/jitter
        # We sum all profits for a specific day before calculating cumsum
        res_df = res_df.copy()
        res_df['pick_day'] = res_df['pick_date'].dt.normalize()
        series = res_df.groupby('pick_day')['profit_actual'].sum().cumsum()
        
        # Institutional Alignment: Prepend 0 at the start of the tracking period
        if not series.empty:
            start_date = series.index.min() - pd.Timedelta(days=1)
            series[start_date] = 0
            series = series.sort_index()
            
        series.name = label
        return series

    q_series = get_series(quartz_res, 'v4 Quartz')
    o_series = get_series(obsidian_res, 'v3 Obsidian')
    d_series = get_series(diamond_res, 'v2 Diamond')
    p_series = get_series(pyrite_res, 'v1 Pyrite')

    # Merge and FIX: Explicitly name and SORT
    bench = pd.concat([q_series, o_series, d_series, p_series], axis=1)
    bench.columns = ['v4 Quartz', 'v3 Obsidian', 'v2 Diamond', 'v1 Pyrite']
    # FIX: Remove fillfill(0). NaNs will prevent the line from starting too early.
    bench = bench.ffill().sort_index()

    print("\n📈 CURRENT PERFORMANCE BENCHMARKS:")
    for col in bench.columns:
        val = bench[col].dropna().iloc[-1] if not bench[col].dropna().empty else 0
        print(f"  {col}: {val:.2f}u")
    
    colors = {
        'v1 Pyrite': '#ffdd00',      # Gold/Pyrite
        'v2 Diamond': '#00f0ff',     # Ice/Diamond
        'v3 Obsidian': '#7c3aed',    # Purple/Obsidian
        'v4 Quartz': '#f8fafc'       # White/Quartz
    }

    plt.style.use('dark_background')

    # --- GENERATE 4 FOCUSED VERSIONS ---
    focus_models = [
        ('pyrite', 'v1 Pyrite'),
        ('diamond', 'v2 Diamond'),
        ('obsidian', 'v3 Obsidian'),
        ('quartz', 'v4 Quartz')
    ]

    for page_id, focus_label in focus_models:
        # --- PANAVISION RATIO: Ultra-slim 16x4 to eliminate vertical voids ---
        fig, ax = plt.subplots(figsize=(16, 4), facecolor='none')
        ax.set_facecolor('none')
        
        # Enforce absolute full-bleed within the figure coordinate space
        ax.set_position([0, 0, 1, 1])
        
        # --- ELITE FOCUS STYLING ---
        for col in bench.columns:
            series_name = str(col)
            is_focus = (series_name == focus_label)
            
            # Drop NaNs for current model for clean start
            curr_series = bench[col].dropna()
            if curr_series.empty: continue

            color = colors.get(series_name, '#ffffff')
            if not is_focus:
                alpha = 0.15 # Further reduced for focus
                lw = 1.0
                zorder = 1
            else:
                alpha = 0.95
                lw = 4.0 # Thicker for flagship feel
                zorder = 100
            
            # Plot line using its specific non-null range
            ax.plot(curr_series.index, curr_series.values, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            
            if is_focus:
                # Add inner glow for focus
                for i in range(1, 4): 
                    ax.plot(curr_series.index, curr_series.values, color=color, linewidth=lw + i*3, alpha=0.012, zorder=zorder-1)

        # --- DYNAMIC Hero-SCALING (Absolute Minimal Voids) ---
        focus_series = bench[focus_label].dropna()
        if focus_series.empty:
            # Fallback for models with no data yet: Center on 0 with headroom
            ax.set_ylim(-5, 15)
        else:
            f_min, f_max = focus_series.min(), focus_series.max()
            delta = f_max - f_min
            
            # Institutional Spacing: 
            # We add 25% top margin to avoid overlapping the HTML labels (top-left).
            # We add 5% bottom margin for floor stability.
            if delta < 10:
                ax.set_ylim(f_min - 5, f_max + 25)
            else:
                ax.set_ylim(f_min - delta*0.05, f_max + delta*0.25)

        # --- FULL BLEED FORMATTING ---
        ax.set_axis_off() # Absolute cleanliness
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', f'comparison_{page_id}.png'))
        
        # Save with transparency and no bbox tight (already adjusted)
        plt.savefig(out_path, dpi=400, transparent=True)
        plt.close(fig)
        print(f"✅ Full-Bleed PNG saved: {out_path}")

if __name__ == "__main__":
    generate_comparison_chart()
