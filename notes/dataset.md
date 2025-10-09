# Dataset
## Dominick's Fine Foods Sales Data
The movement files contain sales information at the store level for each upc in a category. The information is stored on a weekly basis. The files are named wxxx where xxx is the three- letter acronym for the category. Obtain files by UPC code from this download list or by going to Section 5 of this manual.
Remarks
1. **UPC**: This is the key to use while merging with upc file.
2. **Price, Quantity and Movement**: DFF will sometimes bundle products (E.g., 3 cans of tomato soup for $2). In such occasion, the ‘qty’ variable will indicate the size of the bundle (E.g., 3), the price will reflect the total price of the bundle (E.g., $2), but move will reflect the number of actual item sold, not the number of bundles.
Hence, to compute total dollar sales, one must do the following calculation:
Sales = Price * Move / Qty.
3. **Profit**: This variable indicates the gross margin in percent that DFF makes on the sale of the UPC. A profit of 25.3 means that DFF makes 25.3 cents on the dollar for each item sold. This yields a cost of good sold of 74.7 cents.
a) Note however that the wholesale costs in the data do not correspond to replacement cost or the last transaction price. Instead we have the average acquisition cost (AAC) of the items in inventory. This, of course, grates against what economists believe to be the relevant cost for rational decision making.
More precisely, the chain sets retail prices for the next week and also determines AAC at the end of each week, t, according to:
AACt+1 = (Inventory bought in t) Price paidt + (Inventory, end of t-l-salest) AACt
There are two main sources of discrepancy between replacement cost and AAC. The first is the familiar one of sluggish adjustment. A wholesale price cut today only gradually works itself into AAC as old, higher priced inventory is sold off. The second arises from the occasional practice of manufacturers to inform the buyer in advance of an impending temporary price reduction. This permits the buyer to completely deplete inventory and then "overstock" at the lower price. In this case AAC declines precipitously to the lower price and stays there until the large inventory acquired at that price runs off. Thus, the accounting cost shows the low price for some time after the replacement cost has gone back up.
Source: Peltzman, Sam, Prices Rise Faster Than They Fall, Working Paper No. 142, The University of Chicago
4. **Sales**: This variable indicates whether the product was sold on a promotion that week. A code of 'B' indicates a Bonus Buy, 'C' indicates a Coupon, 'S' indicate a simple price reduction. Unfortunately, this variable is not set by DFF on consistent basis (I.e., if the variable is set it indicates a promotion, if it is not set, there might still be a promotion that week).
5. **OK**: This is a flag set by us to indicate that the data for that week are suspect. We do not use flagged data in our analysis.

## Preprocessing
The proprocessing stage will create the following files:

| #  | Category             | Dataset                       | Filename                      | Description                                                                                                  |
|----|----------------------|-------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------|
| 1  | Sales                | Sales                         | sales.csv                     | All cleaned and preprocessed sales data                                                                      |
| 2  | Modeling             | Training Set                  | train.csv                     | Training data for the modeling stage  (~70%)                                                                 |
| 3  | Modeling             | Validation Set                | validation.csv                | Validation data for the modeling stage  (~15%)                                                               |
| 4  | Modeling             | Test Set                      | test.csv                      | Test  data for the modeling stage (~15%)                                                                     |
| 5  | Store Performance    | Same Store Sales Growth       | same_store_sales_growth.csv   | Same store sales growth aggregated by year.                                                                  |
| 6  | Store Performance    | Store Sales Growth            | store_sales_growth.csv        | Growth in sales at the store level for the last year in the training set   vis-à-vis the prior year          |
| 7  | Store Performance    | Store Gross Margin Percent    | store_gross_margin_pct.csv    | Gross margin percent is the ratio of the sum of the gross profit and the   sum of revenue for each store.    |
| 8  | Store Performance    | Store Gross Profit            | store_gross_profit.csv        | Gross profit is the sum of the gross profit at the store level.                                              |
| 9  | Category Performance | Category Sales Growth         | category_sales_growth.csv     | Growth in sales at the category level for the last year in the training   set vis-à-vis the prior year       |
| 10 | Category Performance | Category Gross Margin Percent | category_gross_margin_pct.csv | Gross margin percent is the ratio of the sum of the gross profit and the   sum of revenue for each category. |
| 11 | Category Performance | Category Gross Profit         | category_gross_profit.csv     | Gross profit is the sum of the gross profit at the category level.                                           |

### Sales
Each row will have the following three columns added:
1. **Category**: This is obtained from the file configuration file
2. **Gross Margin Percent**: as follows:
```python
df['GROSS_MARGIN_PCT'] = df['PROFIT'] / 100
```
3. **Gross Profit**: Compute gross profit as follows:
```python
    df['GROSS_PROFIT'] = df['SALES'] * df['GROSS_MARGIN_PERCENT']
```

### Modeling Sets
The training, validation, and test sets will be split temporaly in 70/15/15 splits