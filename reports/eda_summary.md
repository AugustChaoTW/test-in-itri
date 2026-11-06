# AI4I 2020 Predictive Maintenance EDA Summary

**Date**: 2026-11-06  
**Task**: Phase 1 - Exploratory Data Analysis (EDA)  
**Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI ID 601)  
**Source**: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

## 1. Executive Summary

- **Dataset**: 10,000 synthetic industrial machine sensor readings with 14 variables
- **Target**: Binary machine failure indicator (3.39% positive cases)
- **Goal**: Exploratory analysis to understand feature distributions, relationships, and failure patterns for predictive maintenance modeling

## 2. Dataset Overview

| Property | Value |
|----------|-------|
| **Observations** | 10,000 |
| **Features** | 14 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Memory Usage** | ~0.5 MB |

### Variable Types
- **Categorical**: UDI, Product ID, Type (3 levels: L, M, H)
- **Continuous**: 6 sensor measurements
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- **Target**: Machine failure (binary)
- **Failure Mode Indicators**: TWF, HDF, PWF, OSF, RNF (5 independent failure types)

## 3. Target Distribution

### Machine Failure

| Class | Count | Proportion |
|-------|-------|------------|
| 0 (Normal) | 9,661 | 96.61% |
| 1 (Failure) | 339 | 3.39% |

### Failure Mode Breakdown

| Failure Mode | Count | Notes |
|--------------|-------|-------|
| Tool Wear Failure (TWF) | 46 | Occurs after random tool wear 200-240 mins |
| Heat Dissipation Failure (HDF) | 115 | When temp diff < 8.6K & speed < 1380 rpm |
| Power Failure (PWF) | 95 | Power < 3500 W or > 9000 W |
| Overstrain Failure (OSF) | 98 | Tool wear × torque > threshold |
| Random Failure (RNF) | 19 | Chance 0.1% regardless of parameters |

## 4. Product Type Analysis

### Distribution

| Type | Count | Percentage |
|------|-------|------------|
| L (Low) | 6,000 | 60.0% |
| M (Medium) | 2,997 | 30.0% |
| H (High) | 1,003 | 10.0% |

### Failure Rate by Type

| Type | Failure Rate | Notes |
|------|--------------|-------|
| L | 3.92% | Baseline |
| H | 2.09% | Lowest |
| M | 2.77% | Middle |

## 5. Sensor Characteristics

### Descriptive Statistics (All Data)

**Air temperature [K]**: mean=300.005, std=2.000 (range: 295.3-304.5)  
**Process temperature [K]**: mean=310.006, std=1.484 (range: 305.7-313.8)  
**Rotational speed [rpm]**: mean=1538.776, std=179.284 (range: 1168-2886)  
**Torque [Nm]**: mean=39.987, std=9.969 (range: 3.8-76.6)  
**Tool wear [min]**: mean=107.951, std=63.654 (range: 0-253)

### Failure vs Non-Failure Patterns

| Feature | Failure Mean | Normal Mean | Difference |
|---------|--------------|-------------|------------|
| Air temperature | 300.886 | 299.974 | +0.912 |
| Process temperature | 310.290 | 309.996 | +0.294 |
| Rotational speed | 1496.487 | 1540.260 | -43.773 |
| Torque | 50.168 | 39.630 | +10.538 |
| Tool wear | 143.782 | 106.694 | +37.088 |

## 6. Correlation Analysis

### Top Correlations with Machine Failure (Absolute Value)

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| Torque [Nm] | **0.191** | **Strongest positive** |
| Power (W) | **0.176** | **High positive** |
| Tool wear [min] | 0.105 | Moderate positive |
| Air temperature [K] | 0.083 | Weak positive |
| Process temperature [K] | 0.036 | Very weak positive |

**Key Correlations**:
- **Torque** vs Failure: Positive (machines under higher torque more likely to fail)
- **Rotational speed** vs Failure: Negative (-0.044) (slower machines more likely to fail)
- **Temp diff** vs Failure: Negative (-0.112) (smaller temp differences more likely to fail)
- **Rotational speed** vs **Torque**: -0.875 (strong negative relationship)
- **Air temp** vs **Process temp**: 0.876 (very strong relationship)

## 7. Feature Relationships

### Sensor vs Machine Failure (Boxplot View)

- **Torque**: Failure group has significantly higher mean (50.17 vs 39.63)
- **Tool wear**: Failure group shows higher median (165 vs 107)
- **Rotational speed**: Failure group has lower median (1365 vs 1507)
- **Temperature diff**: Failure group has lower values (average -0.11 correlation)

### Derived Features

**Power (W)**: Torque × Rotational speed × π / 30  
- **Correlation with failure**: 0.176 (significant)
- **Business insight**: Machines with power outside 3500-9000 W fail

**Temp diff**: Process temp − Air temp  
- **Correlation with failure**: -0.112 (inverse relationship)
- **Business insight**: Smaller temp differences indicate heat dissipation issues

## 8. Outlier Analysis (IQR Method)

| Feature | Outliers | Notes |
|---------|----------|-------|
| Rotational speed | 418 | ~9% of data, high variance |
| Torque | 69 | ~3% of data, high torque events |
| All other features | 0 | Clean within bounds |

## 9. Key Insights & Recommendations

### 1. Failure Predictability
- **3.39% failure rate** provides a balanced but minority class for classification
- **5 distinct failure modes** suggest multi-output or binary labeling strategies

### 2. Critical Features for Prediction
- **Torque**: Strongest single predictor (r=0.191)
- **Power**: High predictive power, aligns with failure mechanism
- **Temp diff**: Critical for heat-related failures (HDF)

### 3. Operational Recommendations
- **Monitor torque closely** (especially >50 Nm)
- **Track temp differences** (<8.6K threshold for HDF)
- **Watch rotational speed drops** (slowdown may indicate impending failure)

### 4. Modeling Considerations
- **Class imbalance**: Use techniques like SMOTE, class weighting, or anomaly detection
- **Feature engineering**: Include Power and Temp diff as derived features
- **Multi-output potential**: Model five failure types separately or jointly
- **Outlier handling**: Consider robust scaling or outlier removal for speed/torque

## 10. Visualizations

See `notebooks/eda.ipynb` for:
- Histogram distributions of all numerical features
- Failure vs non-failure boxplots
- Correlation heatmap (including derived features)
- Scatter plots of Power and Temp diff vs failure
- Type distribution and failure rate bar charts

## 11. Data Quality Assessment

**Overall Quality**: Excellent  
- Zero missing values
- No duplicate rows
- Consistent physical units
- Synthetic data with realistic distributions
- Well-documented variables and failure mechanisms

**Production Readiness**: ✅ Good for modeling

---

**Report generated automatically by Phase 1 EDA pipeline**
*No manual curation performed - purely data-driven insights*
