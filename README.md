# Grocery Store Performance Analysis & Butcher Block Optimization

A comprehensive data analysis system for grocery store meat and seafood department performance optimization, featuring predictive modeling and automated butcher block space allocation recommendations.

## Overview

This project analyzes grocery store performance across meat and seafood departments to:
- Classify store performance into High, Medium, and Unprofitable categories
- Predict shrink losses using machine learning models
- Optimize butcher block space allocation to maximize profitability
- Generate actionable recommendations for space reduction and cost savings

## Features

### 🏪 Store Performance Analysis
- **Performance Classification**: Categorizes stores based on sales, shrink rates, and labor efficiency
- **Segmentation Analysis**: Groups stores by revenue, efficiency, and shrink control metrics
- **Visual Analytics**: Comprehensive charts and graphs for performance insights

### 🤖 Machine Learning Models
- **Shrink Prediction**: Gradient Boosting Regressor for predicting meat shrink losses
- **Performance Classification**: Multi-class classification for store performance categories
- **Feature Importance**: Identifies key drivers of performance and shrink

### 📏 Butcher Block Optimization
- **Space Allocation**: Recommends optimal butcher block lengths based on performance metrics
- **Cost-Benefit Analysis**: Calculates ROI and payback periods for space changes
- **Sales Impact Modeling**: Projects sales losses under different reduction scenarios

## Requirements

```python
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Requirements

The analysis expects a CSV file named `sample data.csv` with the following columns:

### Required Columns
- `Store ID`: Unique store identifier
- `Division`: Store division/region
- `District #`: District number
- `Banner`: Store banner/brand
- `City`: Store city
- `ST`: State abbreviation

### Performance Metrics
- `Meat Sales`: Annual meat department sales
- `Seafood Sales`: Annual seafood department sales
- `Total Sales`: Total store sales
- `Meat Shrink`: Annual meat shrink losses
- `Seafood Shrink`: Annual seafood shrink losses
- `Meat Dept Shrink Pct`: Meat shrink as percentage of sales
- `Seafood Dept Shrink Pct`: Seafood shrink as percentage of sales
- `Meat Markdown`: Annual meat markdowns
- `Seafood Markdown`: Annual seafood markdowns
- `Meat Markdown Quantity`: Quantity of meat markdowns
- `Seafood Markdown Quantity`: Quantity of seafood markdowns
- `Weekly Labor Hours`: Average weekly labor hours

### Space Metrics
- `Meat Butcher Block`: Current meat butcher block length (feet)
- `Seafood Butcher Block`: Current seafood butcher block length (feet)


## Output

### 1. Performance Classification Report
```
Meat Performance Distribution:
High           45
Medium         123
Unprofitable   87

Seafood Performance Distribution:
High           52
Medium         108
Unprofitable   95
```

### 2. Machine Learning Model Performance
```
=== SHRINK PREDICTION MODEL PERFORMANCE ===
RMSE: $234.56
R²: 0.847
MAE: $189.23

=== CLASSIFICATION MODEL PERFORMANCE ===
Accuracy: 0.923
```

### 3. Butcher Block Optimization Summary
```
=== BUTCHER BLOCK OPTIMIZATION SUMMARY ===
Total_Stores: 255
Stores_with_Changes: 78
High_Priority: 23
Medium_Priority: 31
Total_Cost_Impact: -$45,600
Total_Sales_Impact: -$12,340
```

### 4. Detailed Recommendations
The system generates prioritized recommendations for each store including:
- Current vs. optimal butcher block lengths
- Expected cost savings and sales impact
- Implementation priority (HIGH/MEDIUM/LOW)
- ROI calculations and payback periods

## Key Insights

### Performance Drivers
1. **Sales Volume**: Primary indicator of department success
2. **Shrink Control**: Critical for profitability (target <4-5%)
3. **Labor Efficiency**: Sales per labor hour optimization
4. **Space Utilization**: Sales per square foot of butcher block space

### Optimization Strategy
1. **High Priority**: Stores with >70% below-threshold performance and oversized blocks
2. **Medium Priority**: Stores with 30-70% below-threshold performance
3. **Monitor**: Underperforming stores with minimal space reduction opportunity

### Financial Impact
- Average cost savings: $585 per optimized store
- Typical sales impact: 2-8% reduction in affected departments
- ROI payback period: 6-18 weeks for most recommendations

## Methodology

### Performance Classification Logic
```python
def create_performance_category(sales, shrink_pct, labor_hours, department):
    if department == 'meat':
        if sales > 1800 and shrink_pct < 0.04 and labor_hours > 84:
            return 'High'
        elif sales > 1300 and shrink_pct < 0.05 and labor_hours >= 52:
            return 'Medium'
        else:
            return 'Unprofitable'
```

### Space Optimization Algorithm
1. Calculate performance percentage vs. threshold
2. Determine space reduction based on performance deficit
3. Apply constraints (minimum/maximum lengths)
4. Calculate cost/benefit impact
5. Assign implementation priority

## Visualization Outputs

The analysis generates multiple visualization types:
- Performance distribution charts
- Sales vs. shrink scatter plots with labor hour bubbles
- Feature importance rankings
- Cost vs. sales impact analysis
- Scenario-based sales loss projections

## Model Details

### Gradient Boosting Regressor (Shrink Prediction)
- **Features**: Sales, markdowns, labor hours, location data
- **Target**: Weekly shrink losses
- **Hyperparameters**: 1000 estimators, max_depth=6, learning_rate=0.01

### Gradient Boosting Classifier (Performance Classification)
- **Features**: Sales, shrink metrics, markdowns, labor hours, location data
- **Target**: Performance category (High/Medium/Unprofitable)
- **Accuracy**: Typically >90% on test data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis feature'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create a Pull Request


