# Customer Segmentation for Campaign Optimisation

Unsupervised clustering pipeline that groups retail customers into behavioural segments using KMeans. The segments are then profiled to guide the marketing team in designing targeted promotions for each group.

## Business Context

Marketing campaigns that treat all customers identically tend to underperform. By identifying natural groupings in customer spend, tenure, household composition, and campaign responsiveness, this project provides the marketing team with actionable segment definitions.

## Dataset

`Campaign_data.csv` contains 27 customer-level attributes including demographic information, purchase history across product categories, web behaviour, and responses to previous marketing campaigns.

## Methodology

**Cleaning:** Rows with missing income are dropped (approximately 1% of records). The `Dt_Customer` date field is parsed to datetime.

**Feature Engineering:**
- Age and Age_Range derived from Year_Birth (reference year 2022)
- Customer tenure in days from `Dt_Customer`
- Web conversion rate from web visits and web purchases
- Total spend across all product categories
- Total purchases across all channels
- Household composition (Adults, Dependents, Household_size, Is_Parent)
- Total accepted offers across all campaigns

**Preprocessing:** Categorical columns are label-encoded and all features are standardised with StandardScaler.

**Cluster Selection:** An elbow plot (WCSS vs k) guides the choice of the number of clusters.

**Clustering:** KMeans with the chosen k and PCA 2D projection for visual validation.

**Profiling:** Mean values per cluster across key metrics reveal each segment's defining characteristics.

## Project Structure

```
02_customer_segmentation_campaign/
├── customer_segmentation.py  # End-to-end pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `Campaign_data.csv` in the same directory, then run:

```bash
python customer_segmentation.py
```

Outputs: `elbow_plot.png`, `cluster_plot.png`, and a printed cluster profile table.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CLUSTERS` | 4 | Number of KMeans clusters |
| `reference_year` | 2022 | Year used to compute Age and Customer_For |
