# Exploratory Data Analysis
## Dataset
Present key figures such as:
- Shape of Dataset
- Number of Stores
- Number of Categories
- Number of Weeks
- Count Data
    - Store Count Stats
    - Category Count Stats
    - Week Count Stats

## Performance Analysis

### Store Performance 
#### Same Store Sales Growth
- Compute Same Store Sales Growth YoY for stores open during both the current and previous years.

```python 
# Iterate overall all years starting at zero indexed year 1
for year in (1, n_years):
    # Find the stores that existed during both years
    stores_current = set(df[df['year'] == year]['store_id'].unique())
    stores_previous = set(df[df['year'] == year-1]['store_id'].unique())
    comp_stores = list(stores_current.intersection(stores_previous))

    # Filter the DataFrame to include only these "comp stores"
    df_comp = df[df['store_id'].isin(comp_stores)]
    # Then sum sales for comp_stores for current and previous year, then compute growth.

```
- **Visualization:** Create Bar Chart of Average YoY Sales Growth for the years in the training set

#### Store Sales Growth
**YoY Sales Growth (%):** This is the Same-Store Sales Growth calculation applied to each individual store. It shows which locations are expanding their customer base and sales volume.
To visualize create the following tables:
- Top 10 Stores by YoY Sales Growth (%)    
- Bottom 10 Stores by YoY Sales Growth (%)
#### Store Profitability

#### Store Strategic Quadrant Plot
This immediately segments your stores into four strategic groups:

- **All-Stars (High Growth, High Margin):** The chain's best-run stores.
    
- **Mature & Efficient (Low Growth, High Margin):** Profitable but perhaps market-saturated stores.
    
- **Expanding but Inefficient (High Growth, Low Margin):** Stores that might be using too many promotions to drive growth.
    
- **Underperformers (Low Growth, Low Margin):** The problem stores that require the most attention.

### Seasonality Analysis
With 260 weeks of data, you can and should analyze seasonality.

- Action: Plot the total sales revenue for all stores on a weekly basis for the entire 5-year period. You should be able to see clear, repeating patterns—spikes around major holidays (Thanksgiving, Christmas) and dips in slower months (like January/February).
- Representation: A line chart showing total weekly sales over the 260 weeks is perfect. This adds a layer of professional polish and shows you understand the operational rhythm of a retail business.

### Category Performance
#### Category Growth
The category-level data is a goldmine for understanding growth drivers.

- Action: Group the sales by major categories (e.g., produce, dairy, packaged goods, meat). Calculate the Compound Annual Growth Rate (CAGR) for each category over the 5-year period.
- Representation: Use a bar chart to compare the CAGR of the different categories. This will immediately show which parts of the business are growing, stagnating, or shrinking.

#### Category Profitability
For the last year in the training data, produce a st