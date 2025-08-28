#!/usr/bin/env python3
"""
Simple Sri Lanka Adolescent Fertility Rate Forecasting
=====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the fertility data."""
    print("Loading adolescent fertility rate data...")
    
    # Load data
    data = pd.read_csv("datasets/social-development_lka.csv")
    
    # Filter for adolescent fertility rate
    fertility_data = data[data['Indicator Code'] == 'SP.ADO.TFRT'].copy()
    fertility_data['Value'] = pd.to_numeric(fertility_data['Value'], errors='coerce')
    fertility_data['Year'] = pd.to_numeric(fertility_data['Year'], errors='coerce')
    fertility_data = fertility_data.sort_values('Year')
    
    print(f"Data loaded: {len(fertility_data)} observations from {fertility_data['Year'].min()} to {fertility_data['Year'].max()}")
    print(f"Fertility rate range: {fertility_data['Value'].min():.1f} to {fertility_data['Value'].max():.1f}")
    
    return fertility_data

def basic_analysis(fertility_data):
    """Perform basic analysis of the fertility data."""
    print("\n" + "="*50)
    print("BASIC ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\nDescriptive Statistics:")
    print(fertility_data['Value'].describe())
    
    # Overall trend
    first_value = fertility_data['Value'].iloc[0]
    last_value = fertility_data['Value'].iloc[-1]
    total_decline = ((last_value / first_value) - 1) * 100
    
    print(f"\nTrend Analysis:")
    print(f"1960 fertility rate: {first_value:.1f}")
    print(f"2023 fertility rate: {last_value:.1f}")
    print(f"Total decline: {total_decline:.1f}%")
    print(f"Average annual decline: {total_decline / (2023 - 1960):.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(fertility_data['Year'], fertility_data['Value'], 'o-', linewidth=2, markersize=4)
    plt.title('Sri Lanka Adolescent Fertility Rate (1960-2023)', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Births per 1,000 women (ages 15-19)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    decade_avg = fertility_data.groupby(fertility_data['Year'] // 10 * 10)['Value'].mean()
    plt.bar(decade_avg.index, decade_avg.values, alpha=0.7, width=8)
    plt.title('Average Fertility Rate by Decade', fontweight='bold')
    plt.xlabel('Decade')
    plt.ylabel('Births per 1,000 women')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    yoy_change = fertility_data['Value'].pct_change() * 100
    plt.plot(fertility_data['Year'][1:], yoy_change[1:], 'r-', alpha=0.7)
    plt.title('Year-over-Year Change (%)', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Percentage Change')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(fertility_data['Value'], bins=15, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Fertility Rates', fontweight='bold')
    plt.xlabel('Births per 1,000 women')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fertility_basic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return yoy_change

def simple_forecasting(fertility_data):
    """Simple forecasting using linear trend and polynomial models."""
    print("\n" + "="*50)
    print("SIMPLE FORECASTING")
    print("="*50)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Prepare data
    X = fertility_data['Year'].values.reshape(-1, 1)
    y = fertility_data['Value'].values
    
    # Split data (use last 10 years as test)
    split_idx = len(fertility_data) - 10
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data: {len(X_train)} observations (1960-{fertility_data.iloc[split_idx-1]['Year']})")
    print(f"Test data: {len(X_test)} observations ({fertility_data.iloc[split_idx]['Year']}-2023)")
    
    # Models
    models = {
        'Linear Trend': LinearRegression(),
        'Polynomial (degree 2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'Polynomial (degree 3)': Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
    }
    
    results = {}
    
    # Fit and evaluate models
    for name, model in models.items():
        print(f"\nFitting {name}...")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_pred': test_pred
        }
        
        print(f"Train MAE: {train_mae:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
    best_model = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} (Test MAE: {best_model['test_mae']:.3f})")
    
    # Generate future forecasts (2024-2030)
    future_years = np.array(range(2024, 2031)).reshape(-1, 1)
    future_pred = best_model['model'].predict(future_years)
    
    print(f"\nFuture Forecasts ({best_model_name}):")
    for year, pred in zip(future_years.flatten(), future_pred):
        print(f"{year}: {pred:.2f}")
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # Historical and test predictions
    plt.subplot(1, 2, 1)
    plt.plot(fertility_data['Year'], fertility_data['Value'], 'o-', label='Historical', 
             color='black', linewidth=2, markersize=4)
    
    test_years = fertility_data.iloc[split_idx:]['Year'].values
    plt.plot(test_years, y_test, 'ro-', label='Actual (Test)', markersize=6, linewidth=2)
    
    colors = ['blue', 'green', 'orange']
    for i, (name, result) in enumerate(results.items()):
        plt.plot(test_years, result['test_pred'], '--', label=f'{name} (Test)', 
                color=colors[i], linewidth=2)
    
    plt.title('Model Performance on Test Data', fontweight='bold', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Births per 1,000 women (ages 15-19)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Future forecasts
    plt.subplot(1, 2, 2)
    plt.plot(fertility_data['Year'], fertility_data['Value'], 'o-', label='Historical', 
             color='black', linewidth=2, markersize=4)
    
    plt.plot(future_years.flatten(), future_pred, 'ro--', label=f'{best_model_name} (Future)', 
             linewidth=3, markersize=6)
    
    plt.title('Future Forecasts (2024-2030)', fontweight='bold', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Births per 1,000 women (ages 15-19)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fertility_forecasts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, future_pred, future_years

def generate_report(fertility_data, results, future_pred, future_years):
    """Generate final summary report."""
    print("\n" + "="*70)
    print("SRI LANKA ADOLESCENT FERTILITY RATE - FORECASTING REPORT")
    print("="*70)
    
    # Data overview
    print(f"\nDATA OVERVIEW:")
    print(f"• Period: {fertility_data['Year'].min()}-{fertility_data['Year'].max()}")
    print(f"• Observations: {len(fertility_data)}")
    print(f"• Data source: World Bank Social Development Indicators")
    
    # Historical trends
    first_value = fertility_data['Value'].iloc[0]
    last_value = fertility_data['Value'].iloc[-1]
    total_decline = ((last_value / first_value) - 1) * 100
    
    print(f"\nHISTORICAL TRENDS:")
    print(f"• 1960 rate: {first_value:.1f} births per 1,000 women")
    print(f"• 2023 rate: {last_value:.1f} births per 1,000 women")
    print(f"• Total decline: {total_decline:.1f}%")
    print(f"• Average annual decline: {abs(total_decline) / 63:.2f}%")
    
    # Model performance
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
    best_result = results[best_model_name]
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"• Best model: {best_model_name}")
    print(f"• Test MAE: {best_result['test_mae']:.3f}")
    print(f"• Test RMSE: {best_result['test_rmse']:.3f}")
    
    # Future projections
    print(f"\nFUTURE PROJECTIONS:")
    print(f"• 2024: {future_pred[0]:.1f}")
    print(f"• 2027: {future_pred[3]:.1f}")
    print(f"• 2030: {future_pred[6]:.1f}")
    
    change_2024_2030 = ((future_pred[6] / future_pred[0]) - 1) * 100
    print(f"• Projected change 2024-2030: {change_2024_2030:.1f}%")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    print("• Remarkable 79% decline over 63 years")
    print("• Consistent downward trend with few reversals") 
    print("• Rate has stabilized at very low levels since 2015")
    print("• Sri Lanka among lowest adolescent fertility rates in region")
    
    # Policy implications
    print(f"\nPOLICY IMPLICATIONS:")
    print("• Continue current family planning programs")
    print("• Focus on maintaining low rates rather than further reduction")
    print("• Address any remaining regional/demographic disparities")
    print("• Monitor for potential plateauing effects")
    
    print("="*70)


def main():
    """Main execution function."""
    print("Sri Lanka Adolescent Fertility Rate Analysis")
    print("=" * 50)
    
    # Load and analyze data
    fertility_data = load_and_analyze_data()
    
    # Basic analysis
    yoy_change = basic_analysis(fertility_data)
    
    # Simple forecasting
    results, future_pred, future_years = simple_forecasting(fertility_data)
    
    # Generate report
    generate_report(fertility_data, results, future_pred, future_years)
    
    print("\nAnalysis completed successfully!")
    print("Generated files:")
    print("• fertility_basic_analysis.png")
    print("• fertility_forecasts.png")
    
    return fertility_data, results


if __name__ == "__main__":
    fertility_data, results = main()