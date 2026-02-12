# my-place-of-codes-it-has-a-lot-of-them
my place of codes it has a lot of them which i use  to do my work it is going to get updated at all the time 
# Code Archive

This is the updated version of "code archive.md" with examples added to all sections. For each cheatsheet item (in `code = explanation` format), I've added a brief example usage below it where applicable. For existing code snippets, I've expanded with runnable examples or notes. Sections with visualizations or tips now include sample data assumptions or full runnable code. The file remains organized and concise.

---

## Python Data Manipulation Tips

### Downcasting Numeric Columns for Memory Optimization

```python
# Identify numeric columns for downcasting
for col in f.select_dtypes(include=np.number).columns:
    f[col] = pd.to_numeric(f[col], downcast='float')

print("Memory usage after downcasting:")
print(f.memory_usage(deep=True))
```

**Example Usage** (assuming `f` is a Pandas DataFrame):
```python
import pandas as pd
import numpy as np

# Sample data
data = {'A': np.random.rand(1000), 'B': np.random.randint(0, 100, 1000)}
f = pd.DataFrame(data)
print("Before:", f.memory_usage(deep=True))

# Apply downcasting
for col in f.select_dtypes(include=np.number).columns:
    f[col] = pd.to_numeric(f[col], downcast='float')

print("After:", f.memory_usage(deep=True))
# Output: Shows reduced memory for float columns.
```

### Converting Object Columns to Category for Memory Optimization

```python
# Assuming object_cols is a list of object-type columns
for col in object_cols:
    f[col] = f[col].astype('category')

print("Object columns converted to category type.")
print("Memory usage after conversion:")
print(f.memory_usage(deep=True))
```

**Example Usage**:
```python
import pandas as pd

# Sample data
data = {'Category': ['A', 'B', 'A', 'C'] * 250}
f = pd.DataFrame(data)
object_cols = ['Category']
print("Before:", f.memory_usage(deep=True))

# Apply conversion
for col in object_cols:
    f[col] = f[col].astype('category')

print("After:", f.memory_usage(deep=True))
# Output: Reduced memory for categorical data.
```

---

## Python Visualization Examples

### Violin Plot of Price_USD by Region

```python
plt.figure(figsize=(12, 6))
sns.violinplot(x='Region', y='Price_USD', data=f)
plt.title('Violin Plot of Price_USD by Region')
plt.xlabel('Region')
plt.ylabel('Price (USD)')
plt.show()
```

**Example Usage** (with sample data):
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {'Region': ['North', 'South', 'East', 'West'] * 25,
        'Price_USD': [100, 150, 200, 250] * 25}
f = pd.DataFrame(data)

# Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Region', y='Price_USD', data=f)
plt.title('Violin Plot of Price_USD by Region')
plt.xlabel('Region')
plt.ylabel('Price (USD)')
plt.show()
# Output: Violin plots showing distribution by region.
```

### Jointplot of Price_USD vs. Sales_Volume

```python
plt.figure(figsize=(10, 8))
sns.jointplot(x='Price_USD', y='Sales_Volume', data=f, kind='scatter', height=8, marginal_ticks=True)
plt.suptitle('Jointplot of Price_USD vs. Sales_Volume', y=1.02)
plt.show()
```

**Example Usage**:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {'Price_USD': np.random.uniform(50, 500, 100),
        'Sales_Volume': np.random.uniform(1000, 10000, 100)}
f = pd.DataFrame(data)

# Plot
sns.jointplot(x='Price_USD', y='Sales_Volume', data=f, kind='scatter', height=8, marginal_ticks=True)
plt.suptitle('Jointplot of Price_USD vs. Sales_Volume', y=1.02)
plt.show()
# Output: Scatter with marginal histograms.
```

### Clustermap of Numerical Feature Correlation

```python
# Assuming f_numerical is a DataFrame with numerical columns
correlation_matrix = f_numerical.corr()
sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', figsize=(8, 8))
plt.title('Clustermap of Numerical Feature Correlation')
plt.show()
```

**Example Usage**:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {'A': np.random.rand(100), 'B': np.random.rand(100) + 0.5, 'C': np.random.rand(100)}
f_numerical = pd.DataFrame(data)

# Plot
correlation_matrix = f_numerical.corr()
sns.clustermap(correlation_matrix, annot=True, cmap='coolwarm', figsize=(8, 8))
plt.title('Clustermap of Numerical Feature Correlation')
plt.show()
# Output: Clustered heatmap of correlations.
```

---

## Seaborn Cheatsheet

Seaborn functions for various plot types (as of Seaborn 0.13+ in 2026).

- `sns.scatterplot() = Basic scatter plot with support for hue, size, style semantics`  
  **Example**: `sns.scatterplot(x='x', y='y', data=df, hue='category') # Scatter with color by category.`

- `sns.lineplot() = Line plot (trends, often with error bands via ci/err_style)`  
  **Example**: `sns.lineplot(x='time', y='value', data=df, errorbar='sd', err_style='band') # Line with std dev bands.`

- `sns.relplot() = Figure-level: scatter or line plots, with easy faceting (col/row)`  
  **Example**: `sns.relplot(x='x', y='y', data=df, kind='scatter', col='group') # Faceted scatters.`

- `sns.histplot() = Histogram (univariate or bivariate, with multiple=stack/dodge/etc.)`  
  **Example**: `sns.histplot(data=df, x='value', bins=20, kde=True) # Histogram with KDE overlay.`

- `sns.kdeplot() = Kernel density estimate (univariate, bivariate, filled contours possible)`  
  **Example**: `sns.kdeplot(data=df, x='value', fill=True) # Filled univariate KDE.`

- `sns.ecdfplot() = Empirical CDF plot (great for comparing distributions)`  
  **Example**: `sns.ecdfplot(data=df, x='value', hue='group') # CDF by group.`

- `sns.rugplot() = Marginal rug marks (usually added to other plots)`  
  **Example**: `sns.rugplot(data=df, x='value') # Rug marks on axis.`

- `sns.displot() = Figure-level: hist/kde/ecdf/rug + faceting & combinations`  
  **Example**: `sns.displot(data=df, x='value', kind='kde', col='group') # Faceted KDEs.`

- `sns.stripplot() = Categorical scatter with jitter (dodge possible)`  
  **Example**: `sns.stripplot(x='category', y='value', data=df, jitter=True) # Jittered points.`

- `sns.swarmplot() = Categorical scatter with non-overlapping points (beeswarm style)`  
  **Example**: `sns.swarmplot(x='category', y='value', data=df) # Beeswarm plot.`

- `sns.boxplot() = Box-and-whisker plot (quartiles, outliers, notches option)`  
  **Example**: `sns.boxplot(x='category', y='value', data=df, notch=True) # Boxplot with notches.`

- `sns.boxenplot() = Boxen/variants plot (better for large n, shows more quantiles)`  
  **Example**: `sns.boxenplot(x='category', y='value', data=df) # Enhanced boxplot.`

- `sns.violinplot() = Violin plot (density shape + box/quartiles inside)`  
  **Example**: `sns.violinplot(x='category', y='value', data=df, inner='quartile') # Violin with quartiles.`

- `sns.barplot() = Bar plot (mean/estimator + error bars, can do ci=None)`  
  **Example**: `sns.barplot(x='category', y='value', data=df, errorbar=None) # Mean bars without errors.`

- `sns.countplot() = Bar plot that counts category frequencies`  
  **Example**: `sns.countplot(x='category', data=df) # Frequency bars.`

- `sns.pointplot() = Point estimates (mean) + CI lines (good for trends across categories)`  
  **Example**: `sns.pointplot(x='category', y='value', data=df) # Points with CIs.`

- `sns.catplot() = Figure-level: any categorical kind (box/violin/bar/strip/swarm/point/count) + faceting`  
  **Example**: `sns.catplot(x='category', y='value', data=df, kind='violin', col='group') # Faceted violins.`

- `sns.regplot() = Scatter + single regression fit + CI band`  
  **Example**: `sns.regplot(x='x', y='y', data=df) # Scatter with regression line.`

- `sns.lmplot() = Figure-level: regression across facets (like regplot + col/row/hue)`  
  **Example**: `sns.lmplot(x='x', y='y', data=df, col='group') # Faceted regressions.`

- `sns.heatmap() = 2D heatmap (e.g. correlation matrix, annot=True common)`  
  **Example**: `sns.heatmap(df.corr(), annot=True) # Correlation heatmap.`

- `sns.clustermap() = Clustered heatmap (dendrograms + row/column reordering)`  
  **Example**: `sns.clustermap(df.corr(), cmap='coolwarm') # Clustered correlations.`

- `sns.pairplot() = Pairwise scatter + marginal histograms/KDE (hue supported)`  
  **Example**: `sns.pairplot(df, hue='category') # Pairwise plots by category.`

- `sns.jointplot() = Bivariate center + marginals (kind=scatter/kde/hist/reg/resid/hex)`  
  **Example**: `sns.jointplot(x='x', y='y', data=df, kind='kde') # Joint KDE.`

---

## NumPy Cheatsheet

Common NumPy functions (as of NumPy 2.0+ in 2026).

- `np.array() = Create NumPy array from list, tuple, or nested lists (foundation of everything)`  
  **Example**: `arr = np.array([[1, 2], [3, 4]]) # 2D array.`

- `np.asarray() = Similar to np.array() but avoids copying if input is already an array`  
  **Example**: `arr = np.asarray([1, 2, 3]) # Converts list to array without copy if possible.`

- `np.zeros() = Create array filled with zeros (shape as tuple)`  
  **Example**: `zeros = np.zeros((2, 3)) # 2x3 zero matrix.`

- `np.ones() = Create array filled with ones`  
  **Example**: `ones = np.ones(5) # Array of five 1s.`

- `np.full() = Create array filled with specified value`  
  **Example**: `full = np.full((2, 2), 7) # 2x2 array of 7s.`

- `np.eye() = Create 2D identity matrix`  
  **Example**: `eye = np.eye(3) # 3x3 identity matrix.`

- `np.empty() = Create uninitialized array (fast but contains garbage values)`  
  **Example**: `empty = np.empty(4) # Uninitialized array of 4 elements.`

- `np.arange() = Create evenly spaced values in interval (like range() but returns array)`  
  **Example**: `arange = np.arange(0, 10, 2) # [0, 2, 4, 6, 8].`

- `np.linspace() = Create evenly spaced numbers over specified interval (inclusive endpoint)`  
  **Example**: `lin = np.linspace(0, 1, 5) # [0. , 0.25, 0.5 , 0.75, 1. ].`

- `np.logspace() = Evenly spaced on log scale`  
  **Example**: `log = np.logspace(1, 3, 3) # [10., 100., 1000.].`

- `np.reshape() = Change shape of array without changing data`  
  **Example**: `reshaped = np.arange(6).reshape(2, 3) # [[0,1,2],[3,4,5]].`

- `np.ravel() = Flatten array to 1D (view when possible)`  
  **Example**: `flat = np.ravel([[1,2],[3,4]]) # [1,2,3,4].`

- `np.flatten() = Flatten to 1D (always copy)`  
  **Example**: `flat_copy = np.array([[1,2],[3,4]]).flatten() # [1,2,3,4] (copy).`

- `np.transpose() / .T = Transpose matrix (swap axes)`  
  **Example**: `trans = np.array([[1,2],[3,4]]).T # [[1,3],[2,4]].`

- `np.concatenate() = Join arrays along axis (vstack/hstack are special cases)`  
  **Example**: `concat = np.concatenate(([1,2], [3,4])) # [1,2,3,4].`

- `np.vstack() = Stack arrays vertically (row-wise)`  
  **Example**: `vstack = np.vstack(([1,2], [3,4])) # [[1,2],[3,4]].`

- `np.hstack() = Stack arrays horizontally (column-wise)`  
  **Example**: `hstack = np.hstack(([1,2], [3,4])) # [1,2,3,4].`

- `np.split() = Split array into sub-arrays`  
  **Example**: `split = np.split(np.arange(6), 2) # [array([0,1,2]), array([3,4,5])].`

- `np.sum() = Sum of array elements (axis option)`  
  **Example**: `total = np.sum([[1,2],[3,4]]) # 10.`

- `np.mean() = Arithmetic mean`  
  **Example**: `avg = np.mean([1,2,3]) # 2.0.`

- `np.median() = Median value`  
  **Example**: `med = np.median([1,3,2]) # 2.0.`

- `np.std() = Standard deviation`  
  **Example**: `std = np.std([1,2,3]) # 0.81649658.`

- `np.var() = Variance`  
  **Example**: `var = np.var([1,2,3]) # 0.66666667.`

- `np.min() / np.max() = Minimum / maximum value`  
  **Example**: `min_val = np.min([1,2,3]) # 1; max_val = np.max([1,2,3]) # 3.`

- `np.argmin() / np.argmax() = Index of min / max value`  
  **Example**: `idx = np.argmin([3,1,2]) # 1 (index of 1).`

- `np.sort() = Return sorted copy of array`  
  **Example**: `sorted_arr = np.sort([3,1,2]) # [1,2,3].`

- `np.argsort() = Indices that would sort the array`  
  **Example**: `indices = np.argsort([3,1,2]) # [1,2,0].`

- `np.dot() = Dot product (1D) or matrix multiplication (2D+)`  
  **Example**: `dot = np.dot([1,2], [3,4]) # 11.`

- `np.matmul() / @ = Matrix multiplication (preferred over dot in modern code)`  
  **Example**: `mat = np.matmul([[1,2]], [[3],[4]]) # [[11]].`

- `np.linalg.inv() = Matrix inverse`  
  **Example**: `inv = np.linalg.inv([[1,2],[3,4]]) # Inverse matrix.`

- `np.linalg.det() = Matrix determinant`  
  **Example**: `det = np.linalg.det([[1,2],[3,4]]) # -2.0.`

- `np.linalg.eig() = Eigenvalues and eigenvectors`  
  **Example**: `eigvals, eigvecs = np.linalg.eig([[1,2],[3,4]]) # Eigen decomposition.`

- `np.random.rand() = Uniform random [0,1) in given shape`  
  **Example**: `rand = np.random.rand(3) # [0.1, 0.5, 0.9] (random).`

- `np.random.randn() = Standard normal (Gaussian) random numbers`  
  **Example**: `norm = np.random.randn(3) # Gaussian samples.`

- `np.random.randint() = Random integers in half-open interval`  
  **Example**: `ints = np.random.randint(0, 10, 5) # Random ints [0-9].`

- `np.random.choice() = Random sample from 1D array or int`  
  **Example**: `choice = np.random.choice(['a', 'b', 'c'], 2) # Random ['b', 'a'].`

- `np.random.seed() = Set random seed for reproducibility`  
  **Example**: `np.random.seed(42) # Set seed for consistent randoms.`

- `np.where() = Return elements from array based on condition (vectorized if/else)`  
  **Example**: `where = np.where(np.array([1,2,3]) > 1, 'big', 'small') # ['small', 'big', 'big'].`

- `np.clip() = Clip values to min/max bounds`  
  **Example**: `clipped = np.clip([0,5,10], 2, 8) # [2,5,8].`

- `np.unique() = Return sorted unique elements`  
  **Example**: `uniq = np.unique([1,2,2,3]) # [1,2,3].`

- `np.in1d() = Test membership of 1D array in another (like isin)`  
  **Example**: `mem = np.in1d([1,2,3], [2,4]) # [False, True, False].`

---

## Pandas Cheatsheet

Common Pandas functions (as of Pandas 2.2+ in 2026).

- `pd.DataFrame() = Create DataFrame from dict, list of lists, array, etc.`  
  **Example**: `df = pd.DataFrame({'A': [1,2], 'B': [3,4]}) # Simple DF.`

- `pd.Series() = Create 1D labeled array (single column/row)`  
  **Example**: `s = pd.Series([1,2,3], index=['a','b','c']) # Indexed series.`

- `pd.read_csv() = Read CSV file into DataFrame`  
  **Example**: `df = pd.read_csv('file.csv') # Load from CSV.`

- `pd.read_excel() = Read Excel file/sheet`  
  **Example**: `df = pd.read_excel('file.xlsx', sheet_name='Sheet1') # Load Excel sheet.`

- `pd.read_sql() = Read from SQL query/database`  
  **Example**: `df = pd.read_sql('SELECT * FROM table', conn) # From SQL (conn is connection).`

- `pd.to_csv() = Write DataFrame to CSV`  
  **Example**: `df.to_csv('output.csv', index=False) # Save without index.`

- `pd.to_excel() = Write to Excel`  
  **Example**: `df.to_excel('output.xlsx', sheet_name='Data') # Save to Excel.`

- `df.head() = First 5 rows (or n rows)`  
  **Example**: `df.head(3) # Top 3 rows.`

- `df.tail() = Last 5 rows`  
  **Example**: `df.tail(2) # Bottom 2 rows.`

- `df.sample() = Random row(s)`  
  **Example**: `df.sample(5) # 5 random rows.`

- `df.shape = (rows, columns) tuple`  
  **Example**: `print(df.shape) # (100, 3).`

- `df.info() = Summary: types, non-null counts, memory`  
  **Example**: `df.info() # Prints DF summary.`

- `df.describe() = Basic stats (count/mean/std/min/25%/50%/75%/max) for numeric columns`  
  **Example**: `stats = df.describe() # Stats DataFrame.`

- `df.columns = List of column names`  
  **Example**: `cols = df.columns.tolist() # ['A', 'B'].`

- `df.index = Row index/labels`  
  **Example**: `idx = df.index # RangeIndex or custom.`

- `df.dtypes = Data types of each column`  
  **Example**: `types = df.dtypes # Series of dtypes.`

- `df.loc[] = Label-based indexer (rows/columns by name)`  
  **Example**: `row = df.loc[0] # Row by label.`

- `df.iloc[] = Integer/position-based indexer`  
  **Example**: `cell = df.iloc[0, 1] # First row, second col.`

- `df.at[] = Fast single value access by label`  
  **Example**: `val = df.at[0, 'A'] # Scalar value.`

- `df.iat[] = Fast single value by integer position`  
  **Example**: `val = df.iat[0, 0] # Scalar by position.`

- `df['col'] / df.col = Select single column as Series`  
  **Example**: `col = df['A'] # Series 'A'.`

- `df[['col1','col2']] = Select multiple columns as DataFrame`  
  **Example**: `sub = df[['A', 'B']] # Sub DF.`

- `df.drop() = Remove rows or columns`  
  **Example**: `df.drop('A', axis=1) # Drop column 'A'.`

- `df.rename() = Rename columns or index`  
  **Example**: `df.rename(columns={'A': 'NewA'}) # Rename column.`

- `df.sort_values() = Sort by one or more columns`  
  **Example**: `sorted_df = df.sort_values('A', ascending=False) # Descending sort.`

- `df.sort_index() = Sort by index`  
  **Example**: `df.sort_index() # Sort by row labels.`

- `df.groupby() = Group by column(s) → apply aggregation`  
  **Example**: `grouped = df.groupby('category').mean() # Mean by category.`

- `df.agg() / .aggregate() = Apply one or more aggregations (mean, sum, etc.)`  
  **Example**: `agg = df.agg({'A': 'sum', 'B': 'mean'}) # Custom aggs.`

- `df.apply() = Apply function along axis (row/column)`  
  **Example**: `df.apply(np.sqrt) # Square root element-wise.`

- `df.map() = Apply function element-wise on Series`  
  **Example**: `df['A'].map(lambda x: x*2) # Double series.`

- `df.applymap() = Element-wise on whole DataFrame (older versions)`  
  **Example**: `df.applymap(lambda x: x+1) # Increment all.`

- `df.merge() = SQL-style join (inner/left/right/outer)`  
  **Example**: `merged = df1.merge(df2, on='key') # Inner join.`

- `pd.concat() = Concatenate DataFrames along rows or columns`  
  **Example**: `concat = pd.concat([df1, df2], axis=1) # Column-wise.`

- `df.join() = Join on index or key (simpler than merge sometimes)`  
  **Example**: `joined = df1.join(df2) # Index join.`

- `df.isna() / df.isnull() = Detect missing values (True/False)`  
  **Example**: `missing = df.isna() # Bool DF.`

- `df.notna() = Opposite of isna`  
  **Example**: `valid = df.notna() # Bool DF of non-missing.`

- `df.fillna() = Fill missing values (with value, method=ffill/bfill, etc.)`  
  **Example**: `filled = df.fillna(0) # Replace NaN with 0.`

- `df.dropna() = Remove rows/columns with missing values`  
  **Example**: `clean = df.dropna() # Drop rows with NaN.`

- `df.value_counts() = Count unique values in Series (most frequent first)`  
  **Example**: `counts = df['A'].value_counts() # Frequency series.`

- `df.unique() = Array of unique values in Series`  
  **Example**: `uniques = df['A'].unique() # Array of uniques.`

- `df.nunique() = Number of unique values per column`  
  **Example**: `n = df.nunique() # Series of counts.`

- `df.pivot_table() = Create pivot table (like Excel) with aggregation`  
  **Example**: `pivot = df.pivot_table(values='value', index='category', aggfunc='mean') # Mean pivot.`

- `df.melt() = Unpivot wide → long format`  
  **Example**: `long = df.melt(id_vars='id') # Melt to long.`

- `pd.crosstab() = Compute cross-tabulation (contingency table)`  
  **Example**: `tab = pd.crosstab(df['A'], df['B']) # Contingency table.`

- `df.pivot() = Reshape to wide format (no aggregation)`  
  **Example**: `wide = df.pivot(index='id', columns='category', values='value') # Pivot wide.`

- `df.plot() = Quick pandas plotting (line, bar, hist, scatter, etc.)`  
  **Example**: `df.plot(kind='bar') # Bar plot.`

- `df.plot(kind='bar') = Specific plot types via kind=`  
  **Example**: `df['A'].plot(kind='hist') # Histogram.`

---

## Scikit-Learn Cheatsheet

Common scikit-learn functions (as of scikit-learn 1.5+ in 2026).

- `train_test_split() = Split data into random train/test subsets (stratify, shuffle options)`  
  **Example**: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80/20 split.`

- `StandardScaler() = Standardize features by removing mean & scaling to unit variance`  
  **Example**: `scaler = StandardScaler().fit(X); scaled = scaler.transform(X) # Standardized data.`

- `MinMaxScaler() = Scale features to [0,1] range`  
  **Example**: `scaler = MinMaxScaler().fit_transform(X) # Scaled to [0,1].`

- `OneHotEncoder() = Convert categorical features to one-hot (sparse/dense)`  
  **Example**: `enc = OneHotEncoder().fit_transform([['red'], ['blue']]) # One-hot matrix.`

- `LabelEncoder() = Convert labels to 0..n-1 integers (for target, not features usually)`  
  **Example**: `le = LabelEncoder().fit_transform(['cat', 'dog', 'cat']) # [0,1,0].`

- `SimpleImputer() = Fill missing values (mean/median/most_frequent/constant)`  
  **Example**: `imp = SimpleImputer(strategy='mean').fit_transform(X) # Mean-filled.`

- `Pipeline() = Chain multiple steps (preprocess + model) into one object`  
  **Example**: `pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())]) # Chain.`

- `ColumnTransformer() = Apply different preprocessing to different columns (very powerful)`  
  **Example**: `ct = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(), cat_cols)]) # Mixed preprocess.`

- `make_pipeline() = Shorthand for Pipeline without naming steps`  
  **Example**: `pipe = make_pipeline(StandardScaler(), LogisticRegression()) # Nameless pipeline.`

- `GridSearchCV() = Exhaustive search over parameter grid with cross-validation`  
  **Example**: `gs = GridSearchCV(model, param_grid, cv=5).fit(X, y) # Best params.`

- `RandomizedSearchCV() = Random search over params (faster for large spaces)`  
  **Example**: `rs = RandomizedSearchCV(model, param_dist, n_iter=10).fit(X, y) # Random search.`

- `cross_val_score() = Evaluate model with cross-validation scores`  
  **Example**: `scores = cross_val_score(model, X, y, cv=5) # Array of CV scores.`

- `cross_validate() = More detailed CV results (multiple metrics, fit/score times)`  
  **Example**: `res = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1']) # Dict of results.`

- `.fit() / .transform() / .predict() = Core methods: train, preprocess, make predictions`  
  **Example**: `model.fit(X_train, y_train); preds = model.predict(X_test) # Train and predict.`

- `.score() = Default performance metric (accuracy for classif, R² for regression)`  
  **Example**: `acc = model.score(X_test, y_test) # Accuracy.`

- `classification_report() = Precision/recall/F1/support table`  
  **Example**: `print(classification_report(y_test, preds)) # Report string.`

- `confusion_matrix() = True/False positive/negative counts matrix`  
  **Example**: `cm = confusion_matrix(y_test, preds) # Matrix array.`

- `accuracy_score() = Fraction of correct predictions`  
  **Example**: `acc = accuracy_score(y_test, preds) # Float accuracy.`

- `mean_squared_error() = MSE (regression)`  
  **Example**: `mse = mean_squared_error(y_test, preds) # MSE value.`

- `r2_score() = Coefficient of determination (regression)`  
  **Example**: `r2 = r2_score(y_test, preds) # R² value.`

- `KMeans() = Classic k-means clustering`  
  **Example**: `km = KMeans(n_clusters=3).fit(X); labels = km.labels_ # Cluster labels.`

- `DBSCAN() = Density-based spatial clustering (no need to specify k)`  
  **Example**: `db = DBSCAN(eps=0.5).fit(X); labels = db.labels_ # Clusters and noise.`

- `PCA() = Principal Component Analysis (dimensionality reduction)`  
  **Example**: `pca = PCA(n_components=2).fit_transform(X) # Reduced data.`

- `TSNE() = t-SNE for visualization of high-dim data`  
  **Example**: `tsne = TSNE(n_components=2).fit_transform(X) # 2D embedding.`

- `SVC() / SVR() = Support Vector Classification / Regression`  
  **Example**: `svc = SVC().fit(X, y); preds = svc.predict(X_test) # SVM classification.`

- `RandomForestClassifier() / RandomForestRegressor() = Ensemble of decision trees`  
  **Example**: `rf = RandomForestClassifier(n_estimators=100).fit(X, y) # RF classifier.`

- `GradientBoostingClassifier() / GradientBoostingRegressor() = Classic GB (before XGBoost etc.)`  
  **Example**: `gb = GradientBoostingClassifier().fit(X, y) # GB classifier.`

- `LogisticRegression() = Logistic regression (binary/multiclass)`  
  **Example**: `lr = LogisticRegression().fit(X, y) # Logistic model.`

- `LinearRegression() = Ordinary least squares linear regression`  
  **Example**: `lin = LinearRegression().fit(X, y); preds = lin.predict(X_test) # Linear preds.`

---

## XGBoost Cheatsheet

- `xgb.XGBClassifier() = Scikit-learn style classifier API`  
  **Example**: `model = xgb.XGBClassifier(objective='binary:logistic').fit(X, y) # Binary classifier.`

- `xgb.XGBRegressor() = Scikit-learn style regressor API`  
  **Example**: `model = xgb.XGBRegressor().fit(X, y) # Regressor.`

- `xgb.XGBRanker() = For learning-to-rank tasks`  
  **Example**: `model = xgb.XGBRanker().fit(X, y, group=groups) # Ranking model.`

- `.fit() = Train the model (supports early_stopping_rounds)`  
  **Example**: `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10) # With early stopping.`

- `.predict() = Make predictions (probability with predict_proba for classif)`  
  **Example**: `preds = model.predict(X_test) # Predictions.`

- `.predict_proba() = Class probabilities (for classification)`  
  **Example**: `probs = model.predict_proba(X_test) # Probabilities.`

- `.get_booster() = Access underlying Booster object`  
  **Example**: `booster = model.get_booster() # Booster access.`

- `.feature_importances_ = Built-in feature importance (gain/weight/cover)`  
  **Example**: `importances = model.feature_importances_ # Array of importances.`

- `.plot_importance() = Visualize feature importances`  
  **Example**: `xgb.plot_importance(model) # Importance plot.`

- `.cv() = Built-in cross-validation`  
  **Example**: `results = xgb.cv(params, dtrain, num_boost_round=10, nfold=5) # CV results.`

- `.train() = Low-level training API (with DMatrix)`  
  **Example**: `dtrain = xgb.DMatrix(X, y); model = xgb.train(params, dtrain) # Low-level train.`

- `DMatrix() = XGBoost's optimized data structure (faster than numpy arrays)`  
  **Example**: `dm = xgb.DMatrix(X, label=y) # DMatrix from data.`

- `plot_tree() = Visualize individual boosted trees`  
  **Example**: `xgb.plot_tree(model, num_trees=0) # First tree plot.`

---

## LightGBM Cheatsheet

- `lgb.LGBMClassifier() = Scikit-learn compatible classifier`  
  **Example**: `model = lgb.LGBMClassifier().fit(X, y) # Classifier.`

- `lgb.LGBMRegressor() = Scikit-learn compatible regressor`  
  **Example**: `model = lgb.LGBMRegressor().fit(X, y) # Regressor.`

- `lgb.Dataset() = LightGBM's data container (supports categorical_feature)`  
  **Example**: `ds = lgb.Dataset(X, label=y, categorical_feature=[0,1]) # Dataset.`

- `lgb.train() = Core training function (params dict + Dataset)`  
  **Example**: `model = lgb.train(params, ds, num_boost_round=100) # Train.`

- `lgb.cv() = Cross-validation`  
  **Example**: `cv_res = lgb.cv(params, ds, num_boost_round=100, nfold=5) # CV dict.`

- `.train() / .predict() = On Booster object`  
  **Example**: `booster = lgb.Booster(params=params); booster.train(ds); preds = booster.predict(X) # Booster usage.`

- `.plot_importance() = Plot feature importances`  
  **Example**: `lgb.plot_importance(model) # Importance plot.`

- `.feature_importance() = Get importance scores (split/gain)`  
  **Example**: `imp = model.feature_importance(importance_type='gain') # Scores.`

- `early_stopping() = Callback for early stopping`  
  **Example**: `callbacks = [lgb.early_stopping(stopping_rounds=10)] # In train params.`

- `lgb.plot_metric() = Plot training/validation metrics over iterations`  
  **Example**: `lgb.plot_metric(eval_results) # Metric plot.`

- `lgb.plot_tree() = Visualize a tree`  
  **Example**: `lgb.plot_tree(model, tree_index=0) # Tree plot.`

---

## CatBoost Cheatsheet

- `cb.CatBoostClassifier() = Main classifier (handles categorical natively)`  
  **Example**: `model = cb.CatBoostClassifier().fit(X, y) # Classifier.`

- `cb.CatBoostRegressor() = Main regressor`  
  **Example**: `model = cb.CatBoostRegressor().fit(X, y) # Regressor.`

- `.fit() = Train (cat_features= list of categorical column indices/names)`  
  **Example**: `model.fit(X, y, cat_features=['cat_col']) # With cats.`

- `.predict() = Predictions`  
  **Example**: `preds = model.predict(X_test) # Predictions.`

- `.predict_proba() = Probabilities for classification`  
  **Example**: `probs = model.predict_proba(X_test) # Probabilities.`

- `.get_feature_importance() = Feature importance (PredictionValuesChange default)`  
  **Example**: `imp = model.get_feature_importance() # Importances.`

- `.plot_tree() = Visualize trees`  
  **Example**: `model.plot_tree(tree_idx=0) # Tree plot.`

- `.get_all_params() = See current hyperparameters`  
  **Example**: `params = model.get_all_params() # Dict of params.`

- `cv() = Cross-validation helper`  
  **Example**: `cv_res = cb.cv(pool, params) # CV results.`

- `Pool() = CatBoost data container (good for large data/categorical)`  
  **Example**: `pool = cb.Pool(X, y, cat_features=['cat']) # Pool.`

---

## Polars Cheatsheet (Fast Pandas Alternative)

- `pl.DataFrame() = Create DataFrame (from dict, list, arrow, etc.)`  
  **Example**: `df = pl.DataFrame({'A': [1,2], 'B': [3,4]}) # Simple DF.`

- `pl.read_csv() / pl.read_parquet() = Fast readers (lazy by default with scan_csv/scan_parquet)`  
  **Example**: `df = pl.read_csv('file.csv') # Load CSV.`

- `pl.scan_csv() / pl.scan_parquet() = Lazy mode (query optimization before collect)`  
  **Example**: `lazy = pl.scan_csv('file.csv').collect() # Lazy then eager.`

- `.select() = Select columns / expressions`  
  **Example**: `selected = df.select(pl.col('A')) # Column A.`

- `.with_columns() = Add or replace columns (most common chaining method)`  
  **Example**: `df.with_columns(pl.col('A') * 2) # New column.`

- `.filter() = Row filtering (pl.col("age") > 30)`  
  **Example**: `filtered = df.filter(pl.col('A') > 1) # Rows where A > 1.`

- `.group_by() = Group by one or more columns`  
  **Example**: `grouped = df.group_by('category').agg(pl.col('value').sum()) # Sum by category.`

- `.agg() = Aggregate after group_by (pl.col("sales").sum())`  
  **Example**: `agg = df.group_by('category').agg(pl.sum('value')) # Aggregated.`

- `.join() = SQL-style joins (how="inner/left/right/outer/anti/semi")`  
  **Example**: `joined = df1.join(df2, on='key', how='inner') # Inner join.`

- `.sort() = Sort by column(s)`  
  **Example**: `sorted_df = df.sort('A', descending=True) # Descending sort.`

- `.lazy() = Switch to lazy mode for optimization`  
  **Example**: `lazy_df = df.lazy() # To lazy frame.`

- `.collect() = Execute lazy query and get result`  
  **Example**: `eager = lazy_df.collect() # Execute lazy.`

- `pl.col() = Reference a column for expressions`  
  **Example**: `expr = pl.col('A') + 1 # Expression.`

- `pl.when().then().otherwise() = Vectorized if-else`  
  **Example**: `df.with_columns(pl.when(pl.col('A') > 0).then(1).otherwise(0)) # Conditional column.`

- `pl.concat() = Concatenate DataFrames vertically/horizontally`  
  **Example**: `concat = pl.concat([df1, df2], how='vertical') # Row-wise.`

- `pl.melt() / pl.pivot() = Reshape long ↔ wide`  
  **Example**: `melted = df.melt(id_vars='id') # To long format.`

- `pl.Series() = Single column (less common than in Pandas)`  
  **Example**: `s = pl.Series('vals', [1,2,3]) # Series.`

---

## Solidity Cheatsheet (Ethereum Smart Contracts, v0.8.x+ in 2026)

### Basics & Structure

- `// SPDX-License-Identifier: MIT = Standard comment at top of every file (required for many tools/IDEs like Remix/Hardhat to recognize license; use MIT for open-source)`  
  **Example**: `// SPDX-License-Identifier: MIT // At file top.`

- `pragma solidity ^0.8.20; = Compiler version directive (use caret ^ for minor updates, or exact version like 0.8.26 for production; >=0.8.0 enables safe math by default)`  
  **Example**: `pragma solidity ^0.8.20; // Version lock.`

- `contract MyContract { ... } = Defines a smart contract (like a class in OOP; name should be CapWords/PascalCase; contains state vars, functions, events, etc.)`  
  **Example**: `contract Hello { string public greeting = "Hello"; } // Simple contract.`

- `abstract contract MyAbstract { ... } = Abstract contract (cannot be deployed; used for inheritance/base contracts with unimplemented functions)`  
  **Example**: `abstract contract Base { function unimplemented() virtual; } // Abstract.`

- `interface IMyInterface { ... } = Pure interface (only function signatures; no implementation, no state vars; used for external contract interaction)`  
  **Example**: `interface IERC20 { function transfer(address to, uint256 amount) external; } // ERC20 interface.`

- `library MyLib { ... } = Library (stateless, reusable code; attached via using ... for ... or direct calls; often internal functions)`  
  **Example**: `library Math { function add(uint a, uint b) internal pure returns (uint) { return a + b; } } // Math lib.`

- `using MyLib for uint; = Attaches library functions as methods to a type (e.g. using SafeMath for uint; becomes myUint.add(other))`  
  **Example**: `using Math for uint; // Then: uint x = 1; x.add(2);`

### Data Types & Variables

- `uint256 public myNumber = 42; = Unsigned integer (0 to 2^256-1; most common is uint256; public auto-generates getter)`  
  **Example**: `uint256 public totalSupply = 1000; // Public uint.`

- `int256 private temperature; = Signed integer (negative allowed; less common in contracts)`  
  **Example**: `int256 private balance = -10; // Signed int.`

- `bool isActive = true; = Boolean (true/false; gas-efficient)`  
  **Example**: `bool public paused = false; // Bool flag.`

- `address payable owner; = 20-byte Ethereum address (payable allows sending ETH)`  
  **Example**: `address payable public admin = payable(msg.sender); // Payable address.`

- `bytes32 hashValue; = Fixed 32-byte value (common for keccak256 hashes, keys)`  
  **Example**: `bytes32 public key = keccak256("secret"); // Hash.`

- `string public name = "Alice"; = Dynamic UTF-8 string (expensive; prefer bytes32 when possible)`  
  **Example**: `string public tokenName = "MyToken"; // String.`

- `bytes public data; = Dynamic byte array (cheaper than string for raw data)`  
  **Example**: `bytes public rawData = hex"1234"; // Bytes.`

- `uint8[10] fixedArray; = Fixed-size array (static length; cheaper gas)`  
  **Example**: `uint8[3] public fixed = [1,2,3]; // Fixed array.`

- `uint256[] dynamicArray; = Dynamic array (resizable with .push(), .pop())`  
  **Example**: `uint256[] public nums; function add() { nums.push(4); } // Dynamic.`

- `mapping(address => uint256) public balances; = Key-value store (like hashmap; no iteration; most used for balances/ownership)`  
  **Example**: `mapping(address => uint256) public balances; // Address to balance.`

- `struct User { address addr; uint256 balance; bool isVerified; } = Custom struct (group related data; can be used in arrays/mappings)`  
  **Example**: `struct User { address addr; uint balance; } User public user; // Struct.`

- `enum Status { Pending, Active, Paused, Closed } = Enum (named integer constants; gas-efficient; Status.Active == 1 internally)`  
  **Example**: `enum State { Off, On } State public state = State.On; // Enum.`

### Functions

- `function deposit(uint256 amount) external payable { ... } = Regular function (external = called only from outside; payable = accepts ETH)`  
  **Example**: `function deposit() external payable { balances[msg.sender] += msg.value; } // Payable.`

- `function getBalance() public view returns (uint256) { ... } = View function (reads state, no modify; free when called off-chain)`  
  **Example**: `function getBalance() public view returns (uint256) { return balances[msg.sender]; } // View.`

- `function pureMath(uint a, uint b) public pure returns (uint) { return a + b; } = Pure function (no read/write state; cheapest; deterministic)`  
  **Example**: `function add(uint a, uint b) public pure returns (uint) { return a + b; } // Pure.`

- `constructor(address initialOwner) { owner = initialOwner; } = Constructor (runs once on deploy; can be payable; no returns)`  
  **Example**: `constructor() { owner = msg.sender; } // Init owner.`

- `fallback() external payable { ... } = Fallback (called when no function matches calldata; often for receiving plain ETH)`  
  **Example**: `fallback() external payable { } // Empty fallback.`

- `receive() external payable { ... } = Receive (special fallback only for plain ETH transfers; no calldata; preferred since 0.6+)`  
  **Example**: `receive() external payable { } // Receive ETH.`

### Visibility & State Mutability

- `public = Anyone can call/read (auto-getter for state vars)`  
  **Example**: `uint public x; // Auto getter: function x() external view returns (uint).`

- `external = Only external calls (cheaper for large calldata; cannot be called internally)`  
  **Example**: `function ext() external { } // External only.`

- `internal = Only this contract + children (default for state vars)`  
  **Example**: `function intFunc() internal { } // Internal.`

- `private = Only this contract (not even children; use for sensitive internal logic)`  
  **Example**: `uint private secret; // Private var.`

- `view = Promises not to modify state (can read; cheap off-chain)`  
  **Example**: `function read() public view returns (uint) { return x; } // View.`

- `pure = No state read/write (cheapest; good for math/utils)`  
  **Example**: `function math() public pure returns (uint) { return 1+1; } // Pure.`

- `payable = Can receive ETH (msg.value > 0 allowed)`  
  **Example**: `function pay() public payable { } // Payable.`

### Modifiers

- `modifier onlyOwner() { require(msg.sender == owner, "Not owner"); _; } = Modifier (checks before/after function; _; is placeholder for function body)`  
  **Example**: `modifier onlyOwner() { require(msg.sender == owner); _; } function admin() onlyOwner { } // Usage.`

- `modifier nonReentrant() { require(!locked, "Reentrancy"); locked = true; _; locked = false; } = Reentrancy guard (prevents recursive calls; OpenZeppelin style)`  
  **Example**: `bool locked; modifier nonReentrant() { require(!locked); locked = true; _; locked = false; } // Guard.`

### Error Handling

- `require(balance >= amount, "Insufficient balance"); = Require (reverts with message if false; gas refund up to certain point)`  
  **Example**: `require(msg.sender == owner, "Not owner"); // Check.`

- `revert("Too young"); = Revert (manual revert with reason string)`  
  **Example**: `if (age < 18) revert("Too young"); // Manual revert.`

- `assert(age > 0); = Assert (for invariants; consumes all gas in older versions; now uses Panic since 0.8)`  
  **Example**: `assert(total == a + b); // Invariant check.`

- `custom error InsufficientFunds(uint256 available, uint256 requested); = Custom error (cheaper gas than string; Solidity 0.8.4+)`  
  **Example**: `error InsufficientFunds(uint avail, uint req); // Define.`

- `if (balance < amount) revert InsufficientFunds(balance, amount);`  
  **Example**: `if (balance < amount) revert InsufficientFunds(balance, amount); // Use custom error.`

### Events & Logging

- `event Transfer(address indexed from, address indexed to, uint256 value); = Event declaration (indexed = searchable/filterable off-chain)`  
  **Example**: `event Transfer(address indexed from, address indexed to, uint value); // ERC20 event.`

- `emit Transfer(msg.sender, recipient, amount); = Emit event (logs data to blockchain; cheap way to notify indexers/wallets)`  
  **Example**: `emit Transfer(msg.sender, to, amount); // Emit.`

### Global Variables / Context

- `msg.sender = Caller of current function (address)`  
  **Example**: `address caller = msg.sender; // Current caller.`

- `msg.value = ETH sent with call (in wei)`  
  **Example**: `uint amount = msg.value; // Sent ETH.`

- `msg.data = Calldata bytes`  
  **Example**: `bytes calldata = msg.data; // Input data.`

- `tx.origin = Original sender (dangerous — avoid for auth!)`  
  **Example**: `address orig = tx.origin; // Original tx sender.`

- `block.timestamp = Current block time (approximate; miners can manipulate ±15s)`  
  **Example**: `uint time = block.timestamp; // Now.`

- `block.number = Current block number`  
  **Example**: `uint blk = block.number; // Block height.`

- `block.chainid = Chain ID (helps prevent replay attacks)`  
  **Example**: `uint chain = block.chainid; // e.g., 1 for mainnet.`

- `gasleft() = Remaining gas`  
  **Example**: `uint gas = gasleft(); // Remaining.`

- `address(this).balance = Contract's ETH balance`  
  **Example**: `uint bal = address(this).balance; // Contract balance.`

### Inheritance & Special Keywords

- `contract Child is Parent, Ownable { ... } = Inheritance (multiple allowed; linearization C3)`  
  **Example**: `contract MyToken is ERC20 { } // Inherit ERC20.`

- `super.withdraw(); = Call parent implementation`  
  **Example**: `super.parentFunc(); // Call super.`

- `virtual = Mark function as overridable`  
  **Example**: `function func() virtual { } // Overridable.`

- `override = Required when overriding virtual function`  
  **Example**: `function func() override { } // Override.`

### Other Common Patterns & Keywords

- `immutable uint256 public MAX_SUPPLY; = Immutable (set in constructor; cheaper gas than constant for dynamic values)`  
  **Example**: `immutable uint public MAX_SUPPLY; constructor() { MAX_SUPPLY = 1000; } // Set once.`

- `constant uint256 public FEE = 100; = Constant (compile-time fixed; very cheap)`  
  **Example**: `uint constant FEE = 5; // Fixed fee.`

- `assembly { ... } = Inline assembly (Yul; low-level EVM opcodes; use sparingly for optimization)`  
  **Example**: `assembly { let x := add(1, 2) } // Low-level add.`

- `unchecked { a - b; } = Unchecked block (disable overflow check; gas savings when safe)`  
  **Example**: `unchecked { uint c = a - b; } // No overflow check.`

- `type(MyType).min / .max = Type min/max (e.g. uint256.min == 0)`  
  **Example**: `uint min = uint256.min; // 0.`

---

## Rust Data Mining & ML Cheatsheet (v1.75+ in 2026)

### Common Imports

```rust
use ndarray::prelude::*;              // ≈ numpy — n-dimensional arrays, linear algebra, slicing, broadcasting
// Example: let arr = array![[1., 2.], [3., 4.]]; // 2D array.

use polars::prelude::*;               // ≈ pandas — fastest DataFrame library (often 5–20× faster than pandas), lazy/eager API, CSV/Parquet/JSON/Arrow, group_by, joins, window functions
// Example: let df = DataFrame::new(vec![col("A").i32().into()])?; // Simple DF.

use linfa::prelude::*;                // ≈ scikit-learn core — classical ML algorithms (clustering, regression, classification, decomposition, neighbors, etc.)
// Example: let dataset = Dataset::from((features, targets)); // Dataset.

use linfa_clustering::prelude::*;     // KMeans, DBSCAN, GaussianMixture, etc.
// Example: let clusters = KMeans::params(3).fit(&dataset)?; // Clustering.

use linfa_linear::prelude::*;         // Linear / Ridge / Lasso / ElasticNet regression
// Example: let model = LinearRegression::default().fit(&dataset)?; // Linear reg.

use smartcore::prelude::*;            // alternative / more batteries-included sklearn-like lib (trees, SVM, naive bayes, metrics, model selection…)
// Example: let rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(&x, &y, Default::default()); // RF.

use ndarray_stats::prelude::*;        // basic statistics on ndarray (mean, std, quantiles, covariance…)
// Example: let mean = arr.mean_axis(Axis(0))?; // Column means.

use statrs::prelude::*;               // probability distributions, statistical functions (more mathematical than ndarray_stats)
// Example: let dist = statrs::distribution::Normal::new(0.0, 1.0); // Normal dist.

use plotters::prelude::*;             // ≈ matplotlib — static 2D plotting (line, scatter, histogram, heatmap, contour…)
// Example: let root = BitMapBackend::new("plot.png", (600, 400)).into_drawing_area(); root.fill(&WHITE)?; // Plot setup.

use csv::{ReaderBuilder, WriterBuilder};         // reading/writing CSV (very fast, often used together with polars)
// Example: let mut rdr = ReaderBuilder::new().from_path("file.csv")?; // CSV reader.

use serde::{Serialize, Deserialize};  // JSON / YAML / TOML / CSV serialization (almost every data project needs it)
// Example: #[derive(Serialize, Deserialize)] struct Data { value: i32 }; // Serde struct.

use anyhow::{Result, Context};                   // Clean error handling → fn main() -> Result<()> { … }
// Example: fn func() -> Result<()> { Ok(()) } // Error handling.

use clap::{Parser, Subcommand};                  // CLI argument parsing → very common for tools/scripts
// Example: #[derive(Parser)] struct Args { #[arg(short)] file: String }; let args = Args::parse(); // CLI args.
```

### Deep Learning Additions

```rust
use burn::prelude::*;                 // modern native Rust DL framework (very popular 2025–2026, backends: wgpu, candle, libtorch, ndarray…)
// Example: let tensor = Tensor::<Backend, 2>::from_data(Data::from([[1.0, 2.0], [3.0, 4.0]])); // Tensor.

use candle_core::{Tensor, Device, DType};    // minimal, pytorch-like tensor lib (llama.cpp / candle ecosystem — inference very strong)
// Example: let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?; // Candle tensor.

use ort::{Session, Value};            // ONNX Runtime bindings — best for running pre-trained models (very fast inference)
// Example: let session = Session::builder()?.commit_from_file("model.onnx")?; // ONNX session.
```

---

## Common Python Imports for Data Mining & ML (2026 Edition)

```python
# Data Handling & Visualization
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Better looking defaults (optional but very common)
plt.style.use('seaborn-v0_8')           # or 'ggplot', 'bmh', etc.
sns.set_theme(style="whitegrid", palette="muted")

# Data Preprocessing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    KFold,
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    PolynomialFeatures
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer

# Classical ML Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDClassifier, SGDRegressor
)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay, RocCurveDisplay
)

# Gradient Boosting
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Device setup (common in 2026)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Utilities
import os
import gc
import time
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", "{:,.4f}".format)
np.set_printoptions(precision=4, suppress=True)

print("Common data mining & ML imports loaded ✓")
```

**Example Usage** (full script with sample data):
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Preprocess and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier().fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
print(accuracy_score(y_test, preds))  # Example output: 0.55 (random).
```

---

## Go Cheatsheet (Go 1.22+ in 2026)

### Basics & Structure

- `package main = Defines the package (main for executables; other names for libraries)`  
  **Example**: `package main // Main package.`

- `import "fmt" = Import standard library package (e.g., fmt for printing; multiple imports in parentheses)`  
  **Example**: `import "fmt" // Single import.`

- `import ( "fmt"; "math" ) = Grouped imports for multiple packages`  
  **Example**: `import ( "fmt" "math" ) // Grouped.`

- `func main() { ... } = Entry point for executables (runs automatically)`  
  **Example**: `func main() { fmt.Println("Hello") } // Entry.`

- `const Pi = 3.14 = Constant (compile-time; inferred type or explicit like const Max int = 100)`  
  **Example**: `const Max = 100 // Constant.`

- `var x int = 42 = Variable declaration (global or local; can omit type for inference)`  
  **Example**: `var x int = 42 // Var.`

- `y := "hello" = Short declaration (local only; infers type)`  
  **Example**: `y := "hello" // Short var.`

- `type MyStruct struct { Field1 int; Field2 string } = Define custom struct`  
  **Example**: `type Point struct { X, Y int } // Struct.`

- `type MyInterface interface { Method() string } = Define interface (methods only; no fields)`  
  **Example**: `type Reader interface { Read() string } // Interface.`

### Data Types & Variables

- `int / uint = Signed/unsigned integer (platform-dependent size; use int64/uint64 for fixed)`  
  **Example**: `var i int = 42 // Int.`

- `float64 = Double-precision float (most common; float32 for smaller)`  
  **Example**: `var f float64 = 3.14 // Float.`

- `bool = Boolean (true/false)`  
  **Example**: `var b bool = true // Bool.`

- `string = Immutable UTF-8 string (use backticks for raw literals)`  
  **Example**: `s := "hello" // String.`

- `byte / rune = Alias for uint8 (byte for raw bytes); int32 (rune for Unicode code points)`  
  **Example**: `var b byte = 'a'; var r rune = '😊' // Byte and rune.`

- `[]int = Slice (dynamic array; make([]int, 0, capacity) to initialize)`  
  **Example**: `slice := []int{1,2,3} // Slice literal.`

- `[5]int = Fixed-size array (less common than slices)`  
  **Example**: `arr := [3]int{1,2,3} // Array.`

- `map[string]int = Map (key-value; make(map[string]int) to initialize)`  
  **Example**: `m := make(map[string]int); m["key"] = 1 // Map.`

- `chan int = Channel (for goroutines; make(chan int) or make(chan int, bufferSize))`  
  **Example**: `ch := make(chan int) // Channel.`

- `struct { ... } = Anonymous struct (inline definition)`  
  **Example**: `anon := struct { Name string }{ Name: "Anon" } // Anonymous.`

- `pointer *int = Pointer (use &var for address; *ptr for dereference)`  
  **Example**: `var p *int; i := 42; p = &i; fmt.Println(*p) // 42.`

### Functions

- `func Add(a int, b int) int { return a + b } = Basic function (parameters, return type)`  
  **Example**: `func Add(a, b int) int { return a + b } // Add func.`

- `func MultiReturn() (int, string) { return 42, "ok" } = Multiple returns (common for errors)`  
  **Example**: `func Get() (int, error) { return 1, nil } // Multi return.`

- `func Variadic(args ...int) { ... } = Variadic args (slice of ints; use ... to pass slices)`  
  **Example**: `func Sum(args ...int) int { total := 0; for _, v := range args { total += v }; return total } // Variadic.`

- `func() { ... } = Anonymous function (closures; can capture outer vars)`  
  **Example**: `fn := func(x int) { fmt.Println(x) }; fn(5) // Anon func.`

- `defer func() { ... }() = Defer (runs after function returns; stack-based, great for cleanup)`  
  **Example**: `defer file.Close() // Defer close.`

- `panic("error") / recover() = Panic for errors; recover in defer to handle (like try-catch)`  
  **Example**: `defer func() { if r := recover(); r != nil { fmt.Println("Recovered") } }(); panic("err") // Panic/recover.`

### Control Flow

- `if x > 0 { ... } else if x < 0 { ... } else { ... } = If-else (no parentheses; init stmt optional: if err := fn(); err != nil { ... })`  
  **Example**: `if x > 0 { fmt.Println("positive") } else { fmt.Println("non-positive") } // If.`

- `for i := 0; i < 10; i++ { ... } = Traditional for loop`  
  **Example**: `for i := 0; i < 3; i++ { fmt.Println(i) } // 0 1 2.`

- `for range slice { ... } = Range loop (for i, v := range slice; omit i or v with _)`  
  **Example**: `for i, v := range []int{1,2} { fmt.Println(i, v) } // 0 1; 1 2.`

- `switch x { case 1: ...; default: ... } = Switch (no fallthrough by default; fallthrough keyword optional)`  
  **Example**: `switch x { case 1: fmt.Println("one"); default: fmt.Println("other") } // Switch.`

- `select { case <-ch: ...; case ch <- v: ...; default: ... } = Select for channels (non-blocking multiplex)`  
  **Example**: `select { case v := <-ch: fmt.Println(v); default: fmt.Println("none") } // Select.`

- `goto label = Goto (rare; use sparingly for jumps)`  
  **Example**: `goto End; End: fmt.Println("end") // Goto.`

- `break / continue = Loop control (can label loops for outer control)`  
  **Example**: `for { break } // Break loop.`

### Concurrency (Goroutines & Channels)

- `go func() { ... }() = Start goroutine (lightweight thread; async)`  
  **Example**: `go func() { fmt.Println("async") }() // Goroutine.`

- `ch := make(chan int) = Make channel (unbuffered; buffered with make(chan int, 10))`  
  **Example**: `ch := make(chan int, 1) // Buffered chan.`

- `ch <- value = Send to channel`  
  **Example**: `ch <- 42 // Send.`

- `v := <-ch = Receive from channel (blocks if empty)`  
  **Example**: `v := <-ch // Receive.`

- `close(ch) = Close channel (sender closes; range loop detects)`  
  **Example**: `close(ch) // Close.`

- `sync.Mutex / sync.RWMutex = Mutex for synchronization (mu.Lock(); defer mu.Unlock())`  
  **Example**: `var mu sync.Mutex; mu.Lock(); defer mu.Unlock() // Lock.`

- `sync.WaitGroup = WaitGroup (wg.Add(1); go func() { defer wg.Done(); ... }(); wg.Wait())`  
  **Example**: `var wg sync.WaitGroup; wg.Add(1); go func() { defer wg.Done() }(); wg.Wait() // Wait.`

- `sync.Once = Once (once.Do(func() { ... })) for single execution`  
  **Example**: `var once sync.Once; once.Do(initFunc) // Once.`

- `atomic.AddInt64(&x, 1) = Atomic operations (from sync/atomic; for lock-free)`  
  **Example**: `var x int64; atomic.AddInt64(&x, 1) // Atomic add.`

### Error Handling

- `if err != nil { return err } = Standard error check (most functions return value, error)`  
  **Example**: `val, err := func(); if err != nil { return err } // Check.`

- `errors.New("msg") = Create error (or fmt.Errorf("msg: %v", val))`  
  **Example**: `err := errors.New("failed") // New error.`

- `panic/recover = For unrecoverable errors (use sparingly; prefer errors)`  
  **Example**: `panic("unrecoverable") // Panic.`

- `type MyError struct { ... } ; func (e MyError) Error() string { ... } = Custom error type (implements error interface)`  
  **Example**: `type MyErr struct{}; func (MyErr) Error() string { return "err" } // Custom.`

### Standard Library Highlights

- `fmt.Println("hello") = Print (Printf for formatted; Scan for input)`  
  **Example**: `fmt.Println("hello") // Print.`

- `strings.Split(s, ",") = String ops (Join, Replace, Contains, Trim, etc.)`  
  **Example**: `parts := strings.Split("a,b", ",") // ["a","b"].`

- `time.Now() / time.Sleep(1 * time.Second) = Time (Duration, Timer, Ticker)`  
  **Example**: `now := time.Now() // Current time.`

- `os.Open("file.txt") / ioutil.ReadFile("file.txt") = File I/O (os for low-level; io/fs in 1.16+)`  
  **Example**: `data, _ := os.ReadFile("file.txt") // Read file.`

- `net/http.Get("url") = HTTP client (http.Server for servers)`  
  **Example**: `resp, _ := http.Get("https://example.com") // GET.`

- `json.Marshal(v) / json.Unmarshal(data, &v) = JSON encoding/decoding`  
  **Example**: `jsonData, _ := json.Marshal(map[string]int{"key":1}) // JSON bytes.`

- `log.Fatal("err") = Logging (standard log; or use zap/slog for structured)`  
  **Example**: `log.Println("info") // Log.`

- `testing.T = Tests (func TestX(t *testing.T) { ... }; go test)`  
  **Example**: `func TestAdd(t *testing.T) { if 1+1 != 2 { t.Fail() } } // Test.`

- `flag.Int("port", 8080, "help") = CLI flags (flag.Parse())`  
  **Example**: `port := flag.Int("port", 8080, "port"); flag.Parse() // Flag.`

### Modules & Packages (Go Modules)

- `go mod init example.com/mod = Initialize module`  
  **Example**: `(shell) go mod init mymod // Init.`

- `go get github.com/pkg = Add dependency`  
  **Example**: `(shell) go get github.com/labstack/echo // Get pkg.`

- `go build / go run main.go = Build/run (go install for binaries)`  
  **Example**: `(shell) go run main.go // Run.`

- `go test ./... = Run tests (with -v for verbose)`  
  **Example**: `(shell) go test -v // Tests.`

- `go fmt / go vet = Format/vet code`  
  **Example**: `(shell) go fmt // Format.`

--- 

This updated file now includes examples for all sections and items. Examples are concise, runnable where possible, and demonstrate practical usage. If further adjustments are needed, let me know!
