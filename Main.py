import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
np.random.seed(2024)

data = pd.read_csv('sample data.csv')

meat_cols = ['Store ID', 'Division', 'District #', 'Banner', 'City', 'ST',
             'Meat Sales', 'Total Sales', 'Meat Shrink', 'Total Shrink', 
             'Meat Dept Shrink Pct', 'Meat Markdown', 'Meat Markdown Quantity', 'Weekly Labor Hours']

seafood_cols = ['Store ID', 'Division', 'District #', 'Banner', 'City', 'ST',
                'Seafood Sales', 'Total Sales', 'Seafood Shrink', 'Total Shrink',
                'Seafood Dept Shrink Pct', 'Seafood Markdown', 'Seafood Markdown Quantity', 'Weekly Labor Hours']

meat_data = data[meat_cols].copy()
seafood_data = data[seafood_cols].copy()

character_cols = ['Store ID', 'Division', 'District #', 'Banner', 'City', 'ST']

for col in meat_data.columns:
    if col not in character_cols:
        if 'Sales' in col or 'Shrink' in col or 'Markdown' in col:
            meat_data[col] = pd.to_numeric(meat_data[col], errors='coerce') / 52
        else:
            meat_data[col] = pd.to_numeric(meat_data[col], errors='coerce')

for col in seafood_data.columns:
    if col not in character_cols:
        if 'Sales' in col or 'Shrink' in col or 'Markdown' in col:
            seafood_data[col] = pd.to_numeric(seafood_data[col], errors='coerce') / 52
        else:
            seafood_data[col] = pd.to_numeric(seafood_data[col], errors='coerce')

meat_data = meat_data.dropna(subset=['Meat Sales', 'Meat Dept Shrink Pct', 'Weekly Labor Hours'])
seafood_data = seafood_data.dropna(subset=['Seafood Sales', 'Seafood Dept Shrink Pct', 'Weekly Labor Hours'])

def create_performance_category(row, sales_col, shrink_col):
    if sales_col == 'Meat Sales':
        if row[sales_col] > 1800 and row[shrink_col] < 0.04 and row['Weekly Labor Hours'] > 84:
            return 'High'
        elif row[sales_col] > 1300 and row[shrink_col] < 0.05 and row['Weekly Labor Hours'] >= 52:
            return 'Medium'
        else:
            return 'Unprofitable'
    else:
        if row[sales_col] > 1800 and row[shrink_col] < 0.04 and row['Weekly Labor Hours'] > 84:
            return 'High'
        elif row[sales_col] > 1550 and row[shrink_col] < 0.05 and row['Weekly Labor Hours'] >= 52:
            return 'Medium'
        else:
            return 'Unprofitable'

meat_data['performance_category'] = meat_data.apply(lambda row: create_performance_category(row, 'Meat Sales', 'Meat Dept Shrink Pct'), axis=1)
seafood_data['performance_category'] = seafood_data.apply(lambda row: create_performance_category(row, 'Seafood Sales', 'Seafood Dept Shrink Pct'), axis=1)

print("Meat Performance Distribution:")
print(meat_data['performance_category'].value_counts())
print("\nSeafood Performance Distribution:")
print(seafood_data['performance_category'].value_counts())

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

meat_counts = meat_data['performance_category'].value_counts()
colors = {'High': '#2E7D32', 'Medium': '#1976D2', 'Unprofitable': '#C62828'}
axes[0].bar(meat_counts.index, meat_counts.values, color=[colors[x] for x in meat_counts.index])
axes[0].set_title('Meat Department Performance Distribution')
axes[0].set_ylabel('Number of Stores')

seafood_counts = seafood_data['performance_category'].value_counts()
axes[1].bar(seafood_counts.index, seafood_counts.values, color=[colors[x] for x in seafood_counts.index])
axes[1].set_title('Seafood Department Performance Distribution')
axes[1].set_ylabel('Number of Stores')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=meat_data, x='performance_category', y='Meat Sales', ax=axes[0])
axes[0].set_title('Sales Distribution')
axes[0].set_ylabel('Weekly Sales ($)')

sns.boxplot(data=meat_data, x='performance_category', y='Meat Dept Shrink Pct', ax=axes[1])
axes[1].set_title('Shrink Distribution')
axes[1].set_ylabel('Shrink %')

sns.boxplot(data=meat_data, x='performance_category', y='Weekly Labor Hours', ax=axes[2])
axes[2].set_title('Labor Distribution')
axes[2].set_ylabel('Labor Hours')

plt.tight_layout()
plt.show()

meat_data['Sales_per_Hour'] = meat_data['Meat Sales'] / meat_data['Weekly Labor Hours']

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for category in meat_data['performance_category'].unique():
    subset = meat_data[meat_data['performance_category'] == category]
    axes[0].scatter(subset['Meat Sales'], subset['Meat Dept Shrink Pct'], 
                   s=subset['Weekly Labor Hours']*2, label=category, alpha=0.7, c=colors[category])

axes[0].set_xlabel('Weekly Sales ($)')
axes[0].set_ylabel('Shrink %')
axes[0].set_title('Meat Department: Sales vs Shrink (bubble size = labor hours)')
axes[0].legend()

for category in meat_data['performance_category'].unique():
    subset = meat_data[meat_data['performance_category'] == category]
    axes[1].scatter(subset['Sales_per_Hour'], subset['Meat Dept Shrink Pct'], 
                   label=category, alpha=0.7, c=colors[category])

axes[1].set_xlabel('Sales per Labor Hour ($)')
axes[1].set_ylabel('Shrink %')
axes[1].set_title('Meat Department: Efficiency vs Shrink')
axes[1].legend()

plt.tight_layout()
plt.show()

shrink_features = ['Meat Sales', 'Meat Markdown', 'Meat Markdown Quantity', 'Weekly Labor Hours']
categorical_features = ['Division', 'Banner', 'ST', 'performance_category']

shrink_modeling_data = meat_data[['Store ID', 'Meat Shrink'] + shrink_features + categorical_features].dropna()

le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    shrink_modeling_data[col] = le.fit_transform(shrink_modeling_data[col])
    le_dict[col] = le

X_shrink = shrink_modeling_data.drop(['Store ID', 'Meat Shrink'], axis=1)
y_shrink = shrink_modeling_data['Meat Shrink']

X_train_shrink, X_test_shrink, y_train_shrink, y_test_shrink = train_test_split(
    X_shrink, y_shrink, test_size=0.25, random_state=2024)

scaler_shrink = StandardScaler()
X_train_shrink_scaled = scaler_shrink.fit_transform(X_train_shrink)
X_test_shrink_scaled = scaler_shrink.transform(X_test_shrink)

gbr = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    learning_rate=0.01,
    random_state=2024
)

gbr.fit(X_train_shrink_scaled, y_train_shrink)

y_pred_shrink = gbr.predict(X_test_shrink_scaled)

rmse = np.sqrt(mean_squared_error(y_test_shrink, y_pred_shrink))
r2 = r2_score(y_test_shrink, y_pred_shrink)
mae = np.mean(np.abs(y_test_shrink - y_pred_shrink))

print("=== SHRINK PREDICTION MODEL PERFORMANCE ===")
print(f"RMSE: ${rmse:.2f}")
print(f"R²: {r2:.3f}")
print(f"MAE: ${mae:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_shrink, y_pred_shrink, alpha=0.6)
plt.plot([y_test_shrink.min(), y_test_shrink.max()], [y_test_shrink.min(), y_test_shrink.max()], 'r--', lw=2)
plt.xlabel('Actual Shrink ($)')
plt.ylabel('Predicted Shrink ($)')
plt.title('Actual vs Predicted Meat Shrink')
plt.show()

class_features = ['Meat Sales', 'Meat Shrink', 'Meat Dept Shrink Pct', 'Meat Markdown', 
                 'Meat Markdown Quantity', 'Weekly Labor Hours']
class_categorical = ['Division', 'Banner', 'ST']

class_modeling_data = meat_data[['performance_category'] + class_features + class_categorical].dropna()

le_class_dict = {}
for col in class_categorical:
    le = LabelEncoder()
    class_modeling_data[col] = le.fit_transform(class_modeling_data[col])
    le_class_dict[col] = le

le_target = LabelEncoder()
class_modeling_data['performance_category'] = le_target.fit_transform(class_modeling_data['performance_category'])

X_class = class_modeling_data.drop('performance_category', axis=1)
y_class = class_modeling_data['performance_category']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.25, random_state=2024, stratify=y_class)

scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

gbc = GradientBoostingClassifier(
    n_estimators=1000,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    learning_rate=0.01,
    random_state=2024
)

gbc.fit(X_train_class_scaled, y_train_class)

y_pred_class = gbc.predict(X_test_class_scaled)

accuracy = accuracy_score(y_test_class, y_pred_class)
conf_mat = confusion_matrix(y_test_class, y_pred_class)

print("\n=== CLASSIFICATION MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(conf_mat)

feature_importance_shrink = pd.DataFrame({
    'feature': X_shrink.columns,
    'importance': gbr.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_class = pd.DataFrame({
    'feature': X_class.columns,
    'importance': gbc.feature_importances_
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].barh(feature_importance_shrink['feature'][:10], feature_importance_shrink['importance'][:10])
axes[0].set_title('Top 10 Features for Shrink Prediction')
axes[0].set_xlabel('Importance')

axes[1].barh(feature_importance_class['feature'][:10], feature_importance_class['importance'][:10])
axes[1].set_title('Top 10 Features for Performance Classification')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

meat_data['Sales_per_Hour'] = meat_data['Meat Sales'] / meat_data['Weekly Labor Hours']
meat_data['Shrink_to_Sales'] = meat_data['Meat Shrink'] / meat_data['Meat Sales']
meat_data['Markdown_Rate'] = meat_data['Meat Markdown'] / meat_data['Meat Sales']

meat_data['Revenue_Segment'] = pd.cut(meat_data['Meat Sales'], bins=3, labels=['Low Revenue', 'Mid Revenue', 'High Revenue'])
meat_data['Efficiency_Segment'] = pd.cut(meat_data['Sales_per_Hour'], bins=3, labels=['Low Efficiency', 'Mid Efficiency', 'High Efficiency'])
meat_data['Shrink_Control'] = pd.cut(meat_data['Meat Dept Shrink Pct'], bins=3, labels=['Good Control', 'Average Control', 'Poor Control'])

segment_summary = meat_data.groupby(['Revenue_Segment', 'Shrink_Control']).agg({
    'Store ID': 'count',
    'Meat Sales': 'mean',
    'Meat Dept Shrink Pct': 'mean',
    'Sales_per_Hour': 'mean'
}).round(2)

segment_summary.columns = ['Store_Count', 'Avg_Sales', 'Avg_Shrink_Pct', 'Avg_Efficiency']
print("=== STORE SEGMENTATION SUMMARY ===")
print(segment_summary)

MEAT_BASELINE_LENGTH = 8
SEAFOOD_BASELINE_LENGTH = 12
MIN_MEAT_LENGTH = 4
MIN_SEAFOOD_LENGTH = 6
MAX_MEAT_LENGTH = 16
MAX_SEAFOOD_LENGTH = 20
ADJUSTMENT_INCREMENT = 2
COST_PER_FOOT_MEAT = 150
COST_PER_FOOT_SEAFOOD = 200

butcher_cols = ['Store ID', 'Division', 'District #', 'Banner', 'City', 'ST',
                'Meat Sales', 'Seafood Sales', 'Total Sales',
                'Meat Shrink', 'Seafood Shrink', 'Meat Dept Shrink Pct', 'Seafood Dept Shrink Pct',
                'Meat Butcher Block', 'Seafood Butcher Block',
                'Meat Markdown', 'Seafood Markdown', 'Weekly Labor Hours']

butcher_data = data[butcher_cols].copy()

for col in butcher_data.columns:
    if col not in character_cols:
        if 'Sales' in col or 'Shrink' in col or 'Markdown' in col:
            butcher_data[col] = pd.to_numeric(butcher_data[col], errors='coerce') / 52
        else:
            butcher_data[col] = pd.to_numeric(butcher_data[col], errors='coerce')

butcher_data = butcher_data.dropna(subset=['Meat Butcher Block', 'Seafood Butcher Block', 'Meat Sales', 'Seafood Sales'])
butcher_data = butcher_data[(butcher_data['Meat Butcher Block'] > 0) & (butcher_data['Seafood Butcher Block'] > 0)]

butcher_data['Meat_Sales_per_Foot'] = butcher_data['Meat Sales'] / butcher_data['Meat Butcher Block']
butcher_data['Seafood_Sales_per_Foot'] = butcher_data['Seafood Sales'] / butcher_data['Seafood Butcher Block']
butcher_data['Meat_Shrink_per_Foot'] = butcher_data['Meat Shrink'] / butcher_data['Meat Butcher Block']
butcher_data['Seafood_Shrink_per_Foot'] = butcher_data['Seafood Shrink'] / butcher_data['Seafood Butcher Block']

def create_meat_performance(row):
    if row['Meat Sales'] > 1800 and row['Meat Dept Shrink Pct'] < 0.04 and row['Weekly Labor Hours'] > 84:
        return 'High'
    elif row['Meat Sales'] > 1500 and row['Meat Dept Shrink Pct'] < 0.05 and row['Weekly Labor Hours'] >= 52:
        return 'Medium'
    else:
        return 'Unprofitable'

def create_seafood_performance(row):
    if row['Seafood Sales'] > 2000 and row['Seafood Dept Shrink Pct'] < 0.04 and row['Weekly Labor Hours'] > 84:
        return 'High'
    elif row['Seafood Sales'] > 1850 and row['Seafood Dept Shrink Pct'] < 0.05 and row['Weekly Labor Hours'] >= 52:
        return 'Medium'
    else:
        return 'Unprofitable'

butcher_data['meat_performance'] = butcher_data.apply(create_meat_performance, axis=1)
butcher_data['seafood_performance'] = butcher_data.apply(create_seafood_performance, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for perf in butcher_data['meat_performance'].unique():
    subset = butcher_data[butcher_data['meat_performance'] == perf]
    axes[0].scatter(subset['Meat Butcher Block'], subset['Meat_Sales_per_Foot'], 
                   label=perf, alpha=0.7, c=colors[perf])

axes[0].axvline(x=MEAT_BASELINE_LENGTH, color='red', linestyle='--', label='Baseline (8 ft)')
axes[0].set_xlabel('Butcher Block Length (feet)')
axes[0].set_ylabel('Sales per Foot ($)')
axes[0].set_title('Meat Department: Block Length vs Sales Efficiency')
axes[0].legend()

for perf in butcher_data['seafood_performance'].unique():
    subset = butcher_data[butcher_data['seafood_performance'] == perf]
    axes[1].scatter(subset['Seafood Butcher Block'], subset['Seafood_Sales_per_Foot'], 
                   label=perf, alpha=0.7, c=colors[perf])

axes[1].axvline(x=SEAFOOD_BASELINE_LENGTH, color='red', linestyle='--', label='Baseline (12 ft)')
axes[1].set_xlabel('Butcher Block Length (feet)')
axes[1].set_ylabel('Sales per Foot ($)')
axes[1].set_title('Seafood Department: Block Length vs Sales Efficiency')
axes[1].legend()

plt.tight_layout()
plt.show()

def create_butcher_recommendations(data):
    result = data.copy()
    
    sales_threshold_meat = 1300
    sales_threshold_seafood = 1550
    
    result['Meat_Performance_Pct'] = (result['Meat Sales'] - sales_threshold_meat) / sales_threshold_meat * 100
    result['Seafood_Performance_Pct'] = (result['Seafood Sales'] - sales_threshold_seafood) / sales_threshold_seafood * 100
    
    def calculate_meat_adjustment(row):
        if row['Meat_Performance_Pct'] <= -70 and row['Meat Butcher Block'] > 11:
            return -8
        elif row['Meat_Performance_Pct'] <= -30 and row['Meat Butcher Block'] > 11:
            return -4
        else:
            return 0
    
    def calculate_seafood_adjustment(row):
        if row['Seafood_Performance_Pct'] <= -70 and row['Seafood Butcher Block'] > 15:
            return -8
        elif row['Seafood_Performance_Pct'] <= -30 and row['Seafood Butcher Block'] > 15:
            return -4
        else:
            return 0
    
    result['Meat_Raw_Adjustment'] = result.apply(calculate_meat_adjustment, axis=1)
    result['Seafood_Raw_Adjustment'] = result.apply(calculate_seafood_adjustment, axis=1)
    
    def calculate_optimal_meat_length(row):
        if (row['Meat Butcher Block'] + row['Meat_Raw_Adjustment']) < MIN_MEAT_LENGTH:
            return max(MIN_MEAT_LENGTH, row['Meat Butcher Block'] - 4)
        elif row['Meat_Raw_Adjustment'] < 0:
            return row['Meat Butcher Block'] + (int(abs(row['Meat_Raw_Adjustment']) / 4) * -4)
        elif row['Meat_Raw_Adjustment'] > 0:
            return min(row['Meat Butcher Block'] + (int(np.ceil(row['Meat_Raw_Adjustment'] / 4)) * 4), MAX_MEAT_LENGTH)
        else:
            return row['Meat Butcher Block']
    
    def calculate_optimal_seafood_length(row):
        if (row['Seafood Butcher Block'] + row['Seafood_Raw_Adjustment']) < MIN_SEAFOOD_LENGTH:
            return max(MIN_SEAFOOD_LENGTH, row['Seafood Butcher Block'] - 4)
        elif row['Seafood_Raw_Adjustment'] < 0:
            return row['Seafood Butcher Block'] + (int(abs(row['Seafood_Raw_Adjustment']) / 4) * -4)
        elif row['Seafood_Raw_Adjustment'] > 0:
            return min(row['Seafood Butcher Block'] + (int(np.ceil(row['Seafood_Raw_Adjustment'] / 4)) * 4), MAX_SEAFOOD_LENGTH)
        else:
            return row['Seafood Butcher Block']
    
    result['Meat_Optimal_Length'] = result.apply(calculate_optimal_meat_length, axis=1)
    result['Seafood_Optimal_Length'] = result.apply(calculate_optimal_seafood_length, axis=1)
    
    result['Meat_Length_Change'] = result['Meat_Optimal_Length'] - result['Meat Butcher Block']
    result['Seafood_Length_Change'] = result['Seafood_Optimal_Length'] - result['Seafood Butcher Block']
    
    def get_meat_priority(row):
        if row['Meat_Performance_Pct'] <= -70 and row['Meat Butcher Block'] > 12 and row['Meat_Length_Change'] == -8:
            return 'HIGH'
        elif row['Meat_Performance_Pct'] <= -40 and row['Meat Butcher Block'] > 12 and row['Meat_Length_Change'] == -4:
            return 'HIGH'
        elif row['Meat_Performance_Pct'] <= -40 and row['Meat Butcher Block'] <= 12:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_seafood_priority(row):
        if row['Seafood_Performance_Pct'] <= -70 and row['Seafood Butcher Block'] > 16 and row['Seafood_Length_Change'] == -8:
            return 'HIGH'
        elif row['Seafood_Performance_Pct'] <= -40 and row['Seafood Butcher Block'] > 16 and row['Seafood_Length_Change'] == -4:
            return 'HIGH'
        elif row['Seafood_Performance_Pct'] <= -40 and row['Seafood Butcher Block'] <= 16:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    result['Meat_Priority'] = result.apply(get_meat_priority, axis=1)
    result['Seafood_Priority'] = result.apply(get_seafood_priority, axis=1)
    
    def get_meat_recommendation(row):
        if row['Meat_Length_Change'] == -8:
            return f"DECREASE by 8 feet (2 blocks) - Performance {abs(row['Meat_Performance_Pct']):.1f}% below threshold, store has >{row['Meat Butcher Block']}ft"
        elif row['Meat_Length_Change'] == -4:
            return f"DECREASE by 4 feet (1 block) - Performance {abs(row['Meat_Performance_Pct']):.1f}% below threshold, store has >{row['Meat Butcher Block']}ft"
        elif row['Meat_Performance_Pct'] <= -40 and row['Meat Butcher Block'] <= 12:
            return f"MONITOR - Performance {abs(row['Meat_Performance_Pct']):.1f}% below threshold, but store only has {row['Meat Butcher Block']}ft"
        else:
            return "MAINTAIN current length"
    
    def get_seafood_recommendation(row):
        if row['Seafood_Length_Change'] == -8:
            return f"DECREASE by 8 feet (2 blocks) - Performance {abs(row['Seafood_Performance_Pct']):.1f}% below threshold, store has >{row['Seafood Butcher Block']}ft"
        elif row['Seafood_Length_Change'] == -4:
            return f"DECREASE by 4 feet (1 block) - Performance {abs(row['Seafood_Performance_Pct']):.1f}% below threshold, store has >{row['Seafood Butcher Block']}ft"
        elif row['Seafood_Performance_Pct'] <= -40 and row['Seafood Butcher Block'] <= 16:
            return f"MONITOR - Performance {abs(row['Seafood_Performance_Pct']):.1f}% below threshold, but store only has {row['Seafood Butcher Block']}ft"
        else:
            return "MAINTAIN current length"
    
    result['Meat_Recommendation'] = result.apply(get_meat_recommendation, axis=1)
    result['Seafood_Recommendation'] = result.apply(get_seafood_recommendation, axis=1)
    
    result['Meat_Cost_Change'] = result['Meat_Length_Change'] * COST_PER_FOOT_MEAT
    result['Seafood_Cost_Change'] = result['Seafood_Length_Change'] * COST_PER_FOOT_SEAFOOD
    result['Total_Cost_Change'] = result['Meat_Cost_Change'] + result['Seafood_Cost_Change']
    
    def get_meat_sales_impact(row):
        if row['Meat_Length_Change'] > 0:
            return row['Meat_Length_Change'] * row['Meat_Sales_per_Foot'] * 0.8
        elif row['Meat_Length_Change'] < 0:
            return row['Meat_Length_Change'] * row['Meat_Sales_per_Foot'] * 0.9
        else:
            return 0
    
    def get_seafood_sales_impact(row):
        if row['Seafood_Length_Change'] > 0:
            return row['Seafood_Length_Change'] * row['Seafood_Sales_per_Foot'] * 0.8
        elif row['Seafood_Length_Change'] < 0:
            return row['Seafood_Length_Change'] * row['Seafood_Sales_per_Foot'] * 0.9
        else:
            return 0
    
    result['Estimated_Meat_Sales_Impact'] = result.apply(get_meat_sales_impact, axis=1)
    result['Estimated_Seafood_Sales_Impact'] = result.apply(get_seafood_sales_impact, axis=1)
    result['Total_Sales_Impact'] = result['Estimated_Meat_Sales_Impact'] + result['Estimated_Seafood_Sales_Impact']
    
    def get_weekly_roi(row):
        if abs(row['Total_Cost_Change']) > 0:
            return (row['Total_Sales_Impact'] * 52) / abs(row['Total_Cost_Change'])
        else:
            return 0
    
    result['Weekly_ROI'] = result.apply(get_weekly_roi, axis=1)
    
    def get_priority_score(row):
        if row['Meat_Priority'] == 'HIGH' or row['Seafood_Priority'] == 'HIGH':
            return 'HIGH'
        elif row['Meat_Priority'] == 'MEDIUM' or row['Seafood_Priority'] == 'MEDIUM':
            return 'MEDIUM'
        else:
            return 'NO CHANGE'
    
    result['Priority_Score'] = result.apply(get_priority_score, axis=1)
    
    def get_overall_performance(row):
        if row['Meat_Performance_Pct'] >= 20 and row['Seafood_Performance_Pct'] >= 20:
            return 'High Performer'
        elif row['Meat_Performance_Pct'] >= -20 and row['Seafood_Performance_Pct'] >= -20:
            return 'Stable Performer'
        else:
            return 'Underperformer'
    
    result['Overall_Performance'] = result.apply(get_overall_performance, axis=1)
    
    return result

butcher_recommendations = create_butcher_recommendations(butcher_data)

print(f"Total stores analyzed: {len(butcher_recommendations)}")
print(f"Stores with recommended changes: {sum(butcher_recommendations['Priority_Score'] != 'NO CHANGE')}")
print(f"High priority changes: {sum(butcher_recommendations['Priority_Score'] == 'HIGH')}")

recommendation_summary = {
    'Total_Stores': len(butcher_recommendations),
    'Stores_with_Changes': sum((butcher_recommendations['Meat_Length_Change'] != 0) | (butcher_recommendations['Seafood_Length_Change'] != 0)),
    'Meat_Decreases_4ft': sum(butcher_recommendations['Meat_Length_Change'] == -4),
    'Meat_Decreases_8ft': sum(butcher_recommendations['Meat_Length_Change'] == -8),
    'Seafood_Decreases_4ft': sum(butcher_recommendations['Seafood_Length_Change'] == -4),
    'Seafood_Decreases_8ft': sum(butcher_recommendations['Seafood_Length_Change'] == -8),
    'High_Priority': sum(butcher_recommendations['Priority_Score'] == 'HIGH'),
    'Medium_Priority': sum(butcher_recommendations['Priority_Score'] == 'MEDIUM'),
    'No_Change': sum(butcher_recommendations['Priority_Score'] == 'NO CHANGE'),
    'Total_Cost_Impact': butcher_recommendations['Total_Cost_Change'].sum(),
    'Total_Sales_Impact': butcher_recommendations['Total_Sales_Impact'].sum(),
}

print("=== BUTCHER BLOCK OPTIMIZATION SUMMARY ===")
for key, value in recommendation_summary.items():
    print(f"{key}: {value}")

high_impact_recommendations = butcher_recommendations[
    butcher_recommendations['Priority_Score'].isin(['HIGH', 'MEDIUM'])
].sort_values(['Priority_Score', 'Total_Cost_Change'], ascending=[False, True])

print("\n=== TOP PRIORITY BUTCHER BLOCK RECOMMENDATIONS ===")
top_recommendations = high_impact_recommendations[['Store ID', 'Division', 'Banner', 
                                                  'meat_performance', 'seafood_performance',
                                                  'Meat Butcher Block', 'Meat_Optimal_Length', 
                                                  'Seafood Butcher Block', 'Seafood_Optimal_Length',
                                                  'Total_Cost_Change', 'Total_Sales_Impact', 
                                                  'Priority_Score']].head(15)
print(top_recommendations)

fig, axes = plt.subplots(3, 1, figsize=(12, 15))

meat_changes = butcher_recommendations['Meat_Length_Change']
axes[0].hist([meat_changes[butcher_recommendations['meat_performance'] == p] for p in ['High', 'Medium', 'Unprofitable']], 
            bins=15, alpha=0.7, label=['High', 'Medium', 'Unprofitable'], 
            color=[colors[p] for p in ['High', 'Medium', 'Unprofitable']])
axes[0].set_title('Distribution of Meat Block Length Changes')
axes[0].set_xlabel('Recommended Length Change (feet)')
axes[0].set_ylabel('Number of Stores')
axes[0].legend()

seafood_changes = butcher_recommendations['Seafood_Length_Change']
axes[1].hist([seafood_changes[butcher_recommendations['seafood_performance'] == p] for p in ['High', 'Medium', 'Unprofitable']], 
            bins=15, alpha=0.7, label=['High', 'Medium', 'Unprofitable'],
            color=[colors[p] for p in ['High', 'Medium', 'Unprofitable']])
axes[1].set_title('Distribution of Seafood Block Length Changes')
axes[1].set_xlabel('Recommended Length Change (feet)')
axes[1].set_ylabel('Number of Stores')
axes[1].legend()

priority_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow', 'NO CHANGE': 'gray'}
for priority in butcher_recommendations['Priority_Score'].unique():
    subset = butcher_recommendations[butcher_recommendations['Priority_Score'] == priority]
    axes[2].scatter(subset['Total_Cost_Change'], subset['Total_Sales_Impact'], 
                   label=priority, alpha=0.7, c=priority_colors[priority])

axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Total Cost Change ($)')
axes[2].set_ylabel('Total Sales Impact ($)')
axes[2].set_title('Cost vs Sales Impact of Recommendations')
axes[2].legend()

plt.tight_layout()
plt.show()

def create_detailed_sales_impact(recommendations_data):
    result = recommendations_data.copy()
    
    def calculate_conservative_meat_loss(row):
        if row['Meat_Length_Change'] < 0:
            return abs(row['Meat_Length_Change']) * row['Meat_Sales_per_Foot'] * 0.10
        return 0
    
    def calculate_conservative_seafood_loss(row):
        if row['Seafood_Length_Change'] < 0:
            return abs(row['Seafood_Length_Change']) * row['Seafood_Sales_per_Foot'] * 0.10
        return 0
    
    def calculate_moderate_meat_loss(row):
        if row['Meat_Length_Change'] < 0:
            return abs(row['Meat_Length_Change']) * row['Meat_Sales_per_Foot'] * 0.25
        return 0
    
    def calculate_moderate_seafood_loss(row):
        if row['Seafood_Length_Change'] < 0:
            return abs(row['Seafood_Length_Change']) * row['Seafood_Sales_per_Foot'] * 0.25
        return 0
    
    def calculate_pessimistic_meat_loss(row):
        if row['Meat_Length_Change'] < 0:
            return abs(row['Meat_Length_Change']) * row['Meat_Sales_per_Foot'] * 0.50
        return 0
    
    def calculate_pessimistic_seafood_loss(row):
        if row['Seafood_Length_Change'] < 0:
            return abs(row['Seafood_Length_Change']) * row['Seafood_Sales_per_Foot'] * 0.50
        return 0
    
    result['Conservative_Meat_Loss'] = result.apply(calculate_conservative_meat_loss, axis=1)
    result['Conservative_Seafood_Loss'] = result.apply(calculate_conservative_seafood_loss, axis=1)
    result['Moderate_Meat_Loss'] = result.apply(calculate_moderate_meat_loss, axis=1)
    result['Moderate_Seafood_Loss'] = result.apply(calculate_moderate_seafood_loss, axis=1)
    result['Pessimistic_Meat_Loss'] = result.apply(calculate_pessimistic_meat_loss, axis=1)
    result['Pessimistic_Seafood_Loss'] = result.apply(calculate_pessimistic_seafood_loss, axis=1)
    
    result['Conservative_Weekly_Loss'] = result['Conservative_Meat_Loss'] + result['Conservative_Seafood_Loss']
    result['Moderate_Weekly_Loss'] = result['Moderate_Meat_Loss'] + result['Moderate_Seafood_Loss']
    result['Pessimistic_Weekly_Loss'] = result['Pessimistic_Meat_Loss'] + result['Pessimistic_Seafood_Loss']
    
    result['Conservative_Annual_Loss'] = result['Conservative_Weekly_Loss'] * 52
    result['Moderate_Annual_Loss'] = result['Moderate_Weekly_Loss'] * 52
    result['Pessimistic_Annual_Loss'] = result['Pessimistic_Weekly_Loss'] * 52
    
    def calculate_weekly_cost_savings(row):
        if row['Meat_Length_Change'] < 0 or row['Seafood_Length_Change'] < 0:
            return abs(row['Meat_Length_Change']) * 25 + abs(row['Seafood_Length_Change']) * 35
        return 0
    
    result['Weekly_Cost_Savings'] = result.apply(calculate_weekly_cost_savings, axis=1)
    
    def calculate_conservative_breakeven(row):
        if row['Weekly_Cost_Savings'] > 0 and row['Conservative_Weekly_Loss'] > 0:
            return row['Conservative_Weekly_Loss'] / row['Weekly_Cost_Savings']
        return 0
    
    def calculate_moderate_breakeven(row):
        if row['Weekly_Cost_Savings'] > 0 and row['Moderate_Weekly_Loss'] > 0:
            return row['Moderate_Weekly_Loss'] / row['Weekly_Cost_Savings']
        return 0
    
    def calculate_pessimistic_breakeven(row):
        if row['Weekly_Cost_Savings'] > 0 and row['Pessimistic_Weekly_Loss'] > 0:
            return row['Pessimistic_Weekly_Loss'] / row['Weekly_Cost_Savings']
        return 0
    
    result['Conservative_Breakeven_Weeks'] = result.apply(calculate_conservative_breakeven, axis=1)
    result['Moderate_Breakeven_Weeks'] = result.apply(calculate_moderate_breakeven, axis=1)
    result['Pessimistic_Breakeven_Weeks'] = result.apply(calculate_pessimistic_breakeven, axis=1)
    
    result['Conservative_Sales_Impact_Pct'] = (result['Conservative_Weekly_Loss'] / (result['Meat Sales'] + result['Seafood Sales'])) * 100
    result['Moderate_Sales_Impact_Pct'] = (result['Moderate_Weekly_Loss'] / (result['Meat Sales'] + result['Seafood Sales'])) * 100
    result['Pessimistic_Sales_Impact_Pct'] = (result['Pessimistic_Weekly_Loss'] / (result['Meat Sales'] + result['Seafood Sales'])) * 100
    
    return result

enhanced_recommendations = create_detailed_sales_impact(butcher_recommendations)

sales_impact_summary = enhanced_recommendations[
    (enhanced_recommendations['Meat_Length_Change'] < 0) | (enhanced_recommendations['Seafood_Length_Change'] < 0)
].sort_values(['Priority_Score', 'Moderate_Weekly_Loss'], ascending=[False, False])

print("\n=== DETAILED SALES IMPACT ANALYSIS ===")
sales_impact_display = sales_impact_summary[['Store ID', 'Banner', 'Meat_Length_Change', 'Seafood_Length_Change',
                                            'Moderate_Weekly_Loss', 'Moderate_Annual_Loss', 
                                            'Moderate_Sales_Impact_Pct', 'Priority_Score']].head(15)
print(sales_impact_display)

aggregate_impact = enhanced_recommendations[
    (enhanced_recommendations['Meat_Length_Change'] < 0) | (enhanced_recommendations['Seafood_Length_Change'] < 0)
].groupby('Priority_Score').agg({
    'Store ID': 'count',
    'Conservative_Weekly_Loss': 'sum',
    'Moderate_Weekly_Loss': 'sum',
    'Pessimistic_Weekly_Loss': 'sum',
    'Conservative_Annual_Loss': 'sum',
    'Moderate_Annual_Loss': 'sum',
    'Pessimistic_Annual_Loss': 'sum',
    'Moderate_Sales_Impact_Pct': 'mean'
}).round(2)

print("\n=== AGGREGATE SALES IMPACT BY PRIORITY ===")
print(aggregate_impact)

stores_with_reductions = enhanced_recommendations[
    (enhanced_recommendations['Meat_Length_Change'] < 0) | (enhanced_recommendations['Seafood_Length_Change'] < 0)
]

sales_impact_viz_data = stores_with_reductions[['Store ID', 'Conservative_Weekly_Loss', 
                                               'Moderate_Weekly_Loss', 'Pessimistic_Weekly_Loss', 
                                               'Priority_Score']].head(30)

scenarios = ['Conservative_Weekly_Loss', 'Moderate_Weekly_Loss', 'Pessimistic_Weekly_Loss']
scenario_colors = ['#2E7D32', '#1976D2', '#C62828']

fig, ax = plt.subplots(figsize=(15, 8))

x_pos = np.arange(len(sales_impact_viz_data))
width = 0.25

for i, scenario in enumerate(scenarios):
    ax.bar(x_pos + i*width, sales_impact_viz_data[scenario], width, 
           label=scenario.replace('_Weekly_Loss', ''), color=scenario_colors[i], alpha=0.8)

ax.set_xlabel('Store ID')
ax.set_ylabel('Weekly Sales Loss ($)')
ax.set_title('Weekly Sales Loss by Store and Scenario (Top 30 stores)')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(sales_impact_viz_data['Store ID'], rotation=45)
ax.legend()

plt.tight_layout()
plt.show()

total_impact_summary = stores_with_reductions.agg({
    'Store ID': 'count',
    'Conservative_Weekly_Loss': 'sum',
    'Moderate_Weekly_Loss': 'sum',
    'Pessimistic_Weekly_Loss': 'sum',
    'Conservative_Annual_Loss': 'sum',
    'Moderate_Annual_Loss': 'sum',
    'Pessimistic_Annual_Loss': 'sum',
    'Moderate_Sales_Impact_Pct': 'mean'
})

print("\n=== COMPANY-WIDE SALES IMPACT SUMMARY ===")
print(f"Total stores with recommended reductions: {int(total_impact_summary['Store ID'])}")
print(f"Company-wide weekly sales loss (moderate scenario): ${total_impact_summary['Moderate_Weekly_Loss']:,.2f}")
print(f"Company-wide annual sales loss (moderate scenario): ${total_impact_summary['Moderate_Annual_Loss']:,.2f}")
print(f"Average store impact percentage: {total_impact_summary['Moderate_Sales_Impact_Pct']:.1f}%")

final_recommendations = enhanced_recommendations[
    enhanced_recommendations['Priority_Score'].isin(['HIGH', 'MEDIUM'])
][['Store ID', 'Division', 'Banner', 'City', 'ST', 'meat_performance', 'seafood_performance',
   'Meat Butcher Block', 'Seafood Butcher Block', 'Meat_Length_Change', 'Seafood_Length_Change',
   'Meat Sales', 'Seafood Sales', 'Moderate_Weekly_Loss', 'Moderate_Annual_Loss',
   'Moderate_Sales_Impact_Pct', 'Priority_Score', 'Moderate_Breakeven_Weeks']].sort_values(
    ['Priority_Score', 'Moderate_Weekly_Loss'], ascending=[False, False])

print("\n=== FINAL RECOMMENDATIONS TABLE ===")
detailed_final = final_recommendations.head(25)
print(detailed_final[['Store ID', 'Banner', 'Meat_Length_Change', 'Seafood_Length_Change', 
                     'Moderate_Weekly_Loss', 'Moderate_Annual_Loss', 'Moderate_Sales_Impact_Pct', 
                     'Priority_Score']].round(2))

print(f"\nAnalysis complete. Total stores analyzed: {len(butcher_recommendations)}")
print(f"Stores with recommended changes: {len(final_recommendations)}")
print(f"High priority stores: {sum(final_recommendations['Priority_Score'] == 'HIGH')}")