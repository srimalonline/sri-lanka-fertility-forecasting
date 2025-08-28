#!/usr/bin/env python3
"""
Quick Dataset Analysis for ML Forecasting Opportunities
======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quick_analysis():
    """Quick analysis of the dataset to identify forecasting opportunities."""
    
    # Load the dataset
    df = pd.read_csv("datasets/social-development_lka.csv", skiprows=1)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Analyze indicators
    indicators = df['#indicator+name'].unique()
    print(f"\nTotal unique indicators: {len(indicators)}")
    
    # Analyze each indicator for forecasting potential
    results = []
    
    for indicator in indicators:
        indicator_data = df[df['#indicator+name'] == indicator].copy()
        indicator_data = indicator_data.sort_values('#date+year')
        
        min_year = indicator_data['#date+year'].min()
        max_year = indicator_data['#date+year'].max()
        data_points = len(indicator_data)
        missing_values = indicator_data['#indicator+value+num'].isna().sum()
        valid_points = data_points - missing_values
        
        if valid_points > 0:
            values = indicator_data['#indicator+value+num'].dropna()
            mean_val = values.mean()
            std_val = values.std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            results.append({
                'Indicator': indicator,
                'Min_Year': min_year,
                'Max_Year': max_year,
                'Data_Points': valid_points,
                'Missing': missing_values,
                'Mean': mean_val,
                'Std': std_val,
                'CV': cv,
                'Forecasting_Score': valid_points * (1 if max_year >= 2020 else 0.5) * (1 if cv > 0.01 else 0.1)
            })
    
    # Convert to DataFrame and sort by forecasting potential
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Forecasting_Score', ascending=False)
    
    print(f"\nTop 5 candidates for forecasting:")
    print("=" * 60)
    
    for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"{i}. {row['Indicator']}")
        print(f"   Years: {row['Min_Year']}-{row['Max_Year']} ({row['Data_Points']} points)")
        print(f"   Mean: {row['Mean']:.3f}, Std: {row['Std']:.3f}, CV: {row['CV']:.3f}")
        print(f"   Score: {row['Forecasting_Score']:.1f}\n")
    
    return results_df

def visualize_top_candidate(df, results_df):
    """Visualize the top candidate."""
    
    top_indicator = results_df.iloc[0]['Indicator']
    print(f"Visualizing top candidate: {top_indicator}")
    
    # Get data for top indicator
    indicator_data = df[df['#indicator+name'] == top_indicator].copy()
    indicator_data = indicator_data.sort_values('#date+year')
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(indicator_data['#date+year'], indicator_data['#indicator+value+num'], 
             marker='o', linewidth=2, markersize=4, color='blue')
    
    # Add trend line
    years = indicator_data['#date+year'].values
    values = indicator_data['#indicator+value+num'].values
    z = np.polyfit(years, values, 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), "r--", alpha=0.7, label=f'Trend (slope: {z[0]:.4f})')
    
    plt.title(f"{top_indicator}\nTime Series Analysis", fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forecasting_candidate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'forecasting_candidate.png'")
    
    return top_indicator

if __name__ == "__main__":
    results_df = quick_analysis()
    
    # Load dataset again for visualization
    df = pd.read_csv("datasets/social-development_lka.csv", skiprows=1)
    top_indicator = visualize_top_candidate(df, results_df)
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Best forecasting candidate: {top_indicator}")
    print("This indicator has the best combination of:")
    print("- Sufficient historical data points")
    print("- Recent data availability")
    print("- Meaningful variation for forecasting")