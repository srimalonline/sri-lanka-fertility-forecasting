#!/usr/bin/env python3
"""
Sri Lanka Social Development Dataset Analysis
============================================

This script analyzes the Sri Lanka Social Development dataset to identify
potential forecasting opportunities and assess data quality for time series analysis.

Author: ML Analysis
Date: 2025-08-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_dataset():
    """Load the dataset and perform initial exploration."""
    
    # Load the dataset
    data_path = Path("datasets/social-development_lka.csv")
    
    print("Loading Sri Lanka Social Development Dataset...")
    print("=" * 60)
    
    # Read the dataset, skipping the header comment row
    df = pd.read_csv(data_path, skiprows=1)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def analyze_indicators(df):
    """Analyze the different indicators and their time coverage."""
    
    print("\n" + "=" * 60)
    print("INDICATOR ANALYSIS")
    print("=" * 60)
    
    # Get unique indicators
    indicators = df['#indicator+name'].unique()
    print(f"\nTotal unique indicators: {len(indicators)}")
    
    print("\nAll indicators:")
    for i, indicator in enumerate(indicators, 1):
        print(f"{i:2d}. {indicator}")
    
    # Analyze time coverage for each indicator
    print(f"\n\nTime Coverage Analysis:")
    print("-" * 40)
    
    indicator_analysis = []
    
    for indicator in indicators:
        indicator_data = df[df['#indicator+name'] == indicator].copy()
        
        # Sort by year
        indicator_data = indicator_data.sort_values('#date+year')
        
        # Get basic stats
        min_year = indicator_data['#date+year'].min()
        max_year = indicator_data['#date+year'].max()
        total_years = len(indicator_data)
        year_range = max_year - min_year + 1
        completeness = (total_years / year_range) * 100 if year_range > 0 else 0
        
        # Check for missing values
        missing_values = indicator_data['#indicator+value+num'].isna().sum()
        data_points = len(indicator_data) - missing_values
        
        indicator_analysis.append({
            'Indicator': indicator,
            'Min Year': min_year,
            'Max Year': max_year,
            'Data Points': data_points,
            'Missing Values': missing_values,
            'Year Range': year_range,
            'Completeness %': completeness
        })
        
        print(f"\n{indicator}")
        print(f"  Years: {min_year}-{max_year} ({year_range} year span)")
        print(f"  Data points: {data_points} (missing: {missing_values})")
        print(f"  Completeness: {completeness:.1f}%")
    
    # Convert to DataFrame for better analysis
    analysis_df = pd.DataFrame(indicator_analysis)
    
    return analysis_df

def identify_forecasting_candidates(df, analysis_df):
    """Identify the best indicators for time series forecasting."""
    
    print("\n" + "=" * 60)
    print("FORECASTING CANDIDATE IDENTIFICATION")
    print("=" * 60)
    
    # Criteria for good forecasting candidates:
    # 1. Long time series (at least 20 data points)
    # 2. High completeness (>80%)
    # 3. Recent data (extends to at least 2020)
    # 4. Meaningful variation in values
    
    print("\nCriteria for good forecasting candidates:")
    print("1. At least 20 data points")
    print("2. Data completeness > 80%")
    print("3. Data extends to at least 2020")
    print("4. Meaningful variation in values")
    
    # Filter candidates
    candidates = analysis_df[
        (analysis_df['Data Points'] >= 20) &
        (analysis_df['Completeness %'] > 80) &
        (analysis_df['Max Year'] >= 2020)
    ].copy()
    
    print(f"\nIndicators meeting basic criteria: {len(candidates)}")
    
    # Analyze variation for each candidate
    candidate_details = []
    
    for _, candidate in candidates.iterrows():
        indicator = candidate['Indicator']
        indicator_data = df[df['#indicator+name'] == indicator].copy()
        indicator_data = indicator_data.sort_values('#date+year')
        
        # Remove missing values for variation analysis
        values = indicator_data['#indicator+value+num'].dropna()
        
        if len(values) > 1:
            variation_coeff = values.std() / values.mean() if values.mean() != 0 else 0
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            
            candidate_details.append({
                'Indicator': indicator,
                'Data Points': candidate['Data Points'],
                'Year Range': f"{candidate['Min Year']}-{candidate['Max Year']}",
                'Completeness %': candidate['Completeness %'],
                'Mean Value': mean_val,
                'Min Value': min_val,
                'Max Value': max_val,
                'Std Dev': values.std(),
                'Coeff of Variation': variation_coeff
            })
    
    candidate_df = pd.DataFrame(candidate_details)
    candidate_df = candidate_df.sort_values('Data Points', ascending=False)
    
    print(f"\nTop forecasting candidates:")
    print("-" * 40)
    
    for i, (_, row) in enumerate(candidate_df.head(5).iterrows(), 1):
        print(f"\n{i}. {row['Indicator']}")
        print(f"   Data points: {row['Data Points']}")
        print(f"   Time range: {row['Year Range']}")
        print(f"   Completeness: {row['Completeness %']:.1f}%")
        print(f"   Value range: {row['Min Value']:.3f} - {row['Max Value']:.3f}")
        print(f"   Coefficient of variation: {row['Coeff of Variation']:.3f}")
    
    return candidate_df

def visualize_top_candidates(df, candidate_df):
    """Create visualizations for the top forecasting candidates."""
    
    print("\n" + "=" * 60)
    print("VISUALIZATION OF TOP CANDIDATES")
    print("=" * 60)
    
    # Select top 3 candidates for visualization
    top_candidates = candidate_df.head(3)
    
    fig, axes = plt.subplots(len(top_candidates), 1, figsize=(12, 4*len(top_candidates)))
    
    if len(top_candidates) == 1:
        axes = [axes]
    
    for i, (_, candidate) in enumerate(top_candidates.iterrows()):
        indicator = candidate['Indicator']
        
        # Get data for this indicator
        indicator_data = df[df['#indicator+name'] == indicator].copy()
        indicator_data = indicator_data.sort_values('#date+year')
        
        # Plot time series
        axes[i].plot(indicator_data['#date+year'], indicator_data['#indicator+value+num'], 
                    marker='o', linewidth=2, markersize=4)
        axes[i].set_title(f"{indicator}\n({candidate['Data Points']} data points, "
                         f"{candidate['Completeness %']:.1f}% complete)", fontsize=10)
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = indicator_data['#date+year'].values
        y_numeric = indicator_data['#indicator+value+num'].interpolate().values
        
        if len(x_numeric) > 1:
            z = np.polyfit(x_numeric, y_numeric, 1)
            p = np.poly1d(z)
            axes[i].plot(x_numeric, p(x_numeric), "r--", alpha=0.7, 
                        label=f'Trend (slope: {z[0]:.4f})')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('top_forecasting_candidates.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'top_forecasting_candidates.png'")

def recommend_forecasting_project(candidate_df):
    """Recommend the best forecasting project based on analysis."""
    
    print("\n" + "=" * 60)
    print("FORECASTING PROJECT RECOMMENDATION")
    print("=" * 60)
    
    # Select the best candidate
    best_candidate = candidate_df.iloc[0]
    
    print(f"RECOMMENDED PROJECT: {best_candidate['Indicator']}")
    print("=" * 80)
    
    print(f"\nDataset Characteristics:")
    print(f"- Data points: {best_candidate['Data Points']}")
    print(f"- Time range: {best_candidate['Year Range']}")
    print(f"- Data completeness: {best_candidate['Completeness %']:.1f}%")
    print(f"- Value range: {best_candidate['Min Value']:.3f} - {best_candidate['Max Value']:.3f}")
    print(f"- Variability (CV): {best_candidate['Coeff of Variation']:.3f}")
    
    print(f"\nWhy this is a good forecasting candidate:")
    print("1. Long time series with sufficient historical data")
    print("2. High data completeness with minimal missing values")
    print("3. Recent data available for validation")
    print("4. Meaningful variation showing trends and patterns")
    print("5. Socio-economically relevant for Sri Lanka's development")
    
    # Generate specific recommendations
    generate_project_recommendations(best_candidate['Indicator'])
    
    return best_candidate['Indicator']

def generate_project_recommendations(indicator_name):
    """Generate specific project recommendations for the selected indicator."""
    
    print(f"\n" + "=" * 60)
    print("COMPLETE ML SOLUTION APPROACH")
    print("=" * 60)
    
    print(f"\nPROBLEM STATEMENT:")
    print(f"Develop a machine learning model to forecast '{indicator_name}' for Sri Lanka")
    print("based on historical social development data, enabling government and policy")
    print("makers to make informed decisions about resource allocation and policy interventions.")
    
    print(f"\nFORECASTING OBJECTIVES:")
    print("1. Predict future values of the indicator for the next 3-5 years")
    print("2. Identify long-term trends and seasonal patterns")
    print("3. Provide confidence intervals for predictions")
    print("4. Analyze the impact of potential policy interventions")
    
    print(f"\nDATA PREPROCESSING STEPS:")
    print("1. Data Cleaning:")
    print("   - Handle missing values using interpolation or forward/backward fill")
    print("   - Detect and handle outliers using statistical methods")
    print("   - Ensure data consistency and validate data quality")
    
    print("2. Feature Engineering:")
    print("   - Create time-based features (year, decade indicators)")
    print("   - Generate lag features (previous 1, 2, 3 years)")
    print("   - Calculate rolling averages and trends")
    print("   - Create difference features to capture change rates")
    
    print("3. Data Transformation:")
    print("   - Apply appropriate scaling if needed")
    print("   - Check for stationarity and apply transformations if necessary")
    print("   - Split data into training/validation/test sets")
    
    print(f"\nFORECASTING ALGORITHMS/MODELS:")
    print("1. Statistical Models:")
    print("   - ARIMA (AutoRegressive Integrated Moving Average)")
    print("   - Exponential Smoothing (Holt-Winters)")
    print("   - Seasonal Decomposition")
    
    print("2. Machine Learning Models:")
    print("   - Linear Regression with time features")
    print("   - Random Forest Regressor")
    print("   - Support Vector Regression (SVR)")
    print("   - Gradient Boosting (XGBoost, LightGBM)")
    
    print("3. Deep Learning Models:")
    print("   - LSTM (Long Short-Term Memory) networks")
    print("   - Prophet (Facebook's forecasting tool)")
    print("   - Neural Prophet")
    
    print("4. Ensemble Methods:")
    print("   - Combine multiple models for better predictions")
    print("   - Weighted averaging based on model performance")
    
    print(f"\nEVALUATION METRICS:")
    print("1. Point Accuracy Metrics:")
    print("   - Mean Absolute Error (MAE)")
    print("   - Root Mean Square Error (RMSE)")
    print("   - Mean Absolute Percentage Error (MAPE)")
    print("   - Symmetric MAPE for better handling of zero values")
    
    print("2. Directional Accuracy:")
    print("   - Directional accuracy (up/down predictions)")
    print("   - Trend consistency metrics")
    
    print("3. Interval Forecasting:")
    print("   - Prediction interval coverage probability")
    print("   - Interval width analysis")
    
    print("4. Model Validation:")
    print("   - Walk-forward validation")
    print("   - Cross-validation for time series")
    print("   - Out-of-sample testing")
    
    print(f"\nIMPLEMENTATION PLAN:")
    print("Phase 1: Data Preparation (Week 1)")
    print("- Load and clean the dataset")
    print("- Perform comprehensive EDA")
    print("- Handle missing values and outliers")
    print("- Create feature engineering pipeline")
    
    print("Phase 2: Model Development (Weeks 2-3)")
    print("- Implement baseline statistical models")
    print("- Develop machine learning models")
    print("- Experiment with deep learning approaches")
    print("- Tune hyperparameters for each model")
    
    print("Phase 3: Model Evaluation and Selection (Week 4)")
    print("- Compare models using multiple metrics")
    print("- Perform walk-forward validation")
    print("- Select best performing model(s)")
    print("- Create ensemble if beneficial")
    
    print("Phase 4: Production and Deployment (Week 5)")
    print("- Create prediction pipeline")
    print("- Develop visualization dashboards")
    print("- Document model assumptions and limitations")
    print("- Prepare final forecasts with confidence intervals")
    
    print(f"\nEXPECTED DELIVERABLES:")
    print("1. Comprehensive data analysis report")
    print("2. Trained forecasting models with performance metrics")
    print("3. Future predictions with confidence intervals")
    print("4. Interactive visualization dashboard")
    print("5. Model documentation and deployment guide")
    print("6. Policy recommendations based on forecasts")

def main():
    """Main analysis function."""
    
    print("Sri Lanka Social Development Dataset Analysis")
    print("=" * 60)
    print("Analyzing datasets for ML forecasting opportunities...")
    
    # Load and explore dataset
    df = load_and_explore_dataset()
    
    # Analyze indicators
    analysis_df = analyze_indicators(df)
    
    # Identify forecasting candidates
    candidate_df = identify_forecasting_candidates(df, analysis_df)
    
    # Create visualizations
    if len(candidate_df) > 0:
        visualize_top_candidates(df, candidate_df)
        
        # Recommend best project
        recommended_indicator = recommend_forecasting_project(candidate_df)
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Recommended forecasting project: {recommended_indicator}")
        print("See 'top_forecasting_candidates.png' for visualizations")
        
    else:
        print("\nNo suitable forecasting candidates found based on the criteria.")
        print("Consider relaxing the requirements or exploring other approaches.")

if __name__ == "__main__":
    main()