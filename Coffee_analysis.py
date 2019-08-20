import csv
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

# Load csv file and read number of lines 
with open('arabica_data_cleaned.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1 
        else:
            line_count += 1
    print('Processed {} lines.'.format(line_count))

df = pd.read_csv('arabica_data_cleaned.csv')

#print(tabulate(df.describe(), headers='keys'))
#print(df.columns)

# drop non-coffee quality columns
df = df[['Country.of.Origin', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity',
          'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 
          'Cupper.Points', 'Total.Cup.Points']]

df = df.rename(columns={'Country.of.Origin':'Country'})
print(df.describe())
#print(df.isnull().sum())
df.drop(df.tail(1).index,inplace=True) # last row is all 0 rating
df = df.dropna() 

""" When looking at the coffee qualities, it seems we are working with 
continuous data. Quality range values are scored from 0 to 10, but only a rating
of 10 were given to categories: Uniformity, Clean Cup, Sweetness and Cupper 
points. 

Aroma, Flavor, Aftertaste, Uniformity, Clean Cup, Sweetness, and Cupper points
have means less than median which suggests a left-skewed distribution for each 
these categories. """

# Check for correlations
corr = df.corr() # Pearson correlation
hmap_labels = df.columns[1:]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, xticklabels=hmap_labels, yticklabels=hmap_labels, annot=True)
plt.xticks(rotation=-60)
plt.tight_layout()


""" Uniformity, Clean Cup, and Sweetness have the lowest correlation values as 
shown in heatmap, while all other categories are shown to have high correlation.
These correlation values agree with data that shows majority of uniformity, 
clean cup, and sweetness ratings are less variable and rated 10/10, while the 
other categories have more variable ratings. """

# Check for outliers in each category
columns = df.columns[1:-1].values

boxplot_data = []
for each in columns:
    boxplot_data.append(df[each])

fig, ax = plt.subplots(figsize=(12, 7))
ax.boxplot(boxplot_data)
ax.set_title('Overall Coffee Quality Dataset')
ax.set_ylabel('Rating')
ax.set_xticklabels(labels=columns)

# Check distribution of values for each category
fig, ax = plt.subplots(2, 5, figsize=(16, 8), sharex=False, sharey=False)
.set_title('All Coffee Quality Distribution')
for i in range(1,11):
    fig.add_subplot(2, 5, i)
    sns.distplot(boxplot_data[i-1], kde=True)
    plt.tight_layout()
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(columns[i-1], labelpad=20)
    
"""
Since there are a lot of outliers in the values for each category, median would 
be a better metric than average. Many of the categories are left-skewed, except
for Uniformity, Clean Cup, and Sweetness"""

# Which are the top 10 countries with the most reviews?
df['Country'].replace('Tanzania, United Republic Of', 'Tanzania', inplace=True)
df_barchart = pd.DataFrame(df['Country'].value_counts())
df_barchart.reset_index(level=0, inplace=True)
df_barchart.columns = ['Country', 'Number of Reviews']
df_barchart = df_barchart[df_barchart['Number of Reviews'] >= 10]

# frequency barchart
x = df_barchart['Country']
y = df_barchart['Number of Reviews']

fig, ax1 = plt.subplots(figsize=(10, 5), tight_layout=True) 
ax1.bar(x, y, width=0.8)
ax1.set_ylabel('Number of Reviews')
ax1.set_xlabel('Countries')
ax1.set_title('Coffee Suppliers Around the World')
plt.xticks(x, df_barchart['Country'], rotation=90, fontsize=10)

"""
Country             Reviews
Mexico                  236
Colombia                183
Guatemala               181
Brazil                  132
Taiwan                   75
United States (Hawaii)   73
Honduras                 53
Costa Rica               51
Ethiopia                 44
Tanzania                 40
"""

# Which country (from top 10 suppliers) has the best quality coffee?
top_10 = df_barchart['Country'][0:10]
country_dict = {}

df_median = pd.DataFrame(None)
for country in top_10:
    country_dict[country] = df[df['Country'] == country]

    country_quality = []
    for col in columns:
        country_quality.append(country_dict[country][col])

    # box plot of coffee quality for each country
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(country_quality)
    ax.set_title('{} Coffee Quality'.format(country))
    ax.set_ylabel('Rating')
    ax.set_xticklabels(labels=columns)
    
    df_median[country] = country_dict[country].median()

df_median = df_median.T
print(df_median)

for col in columns:
    print(col)
    print(df_median[col].idxmax())
    print()

"""
Aroma - Ethiopia
Flavor -Ethiopia
Aftertaste - Ethiopia
Acidity - Ethiopia
Body - Ethiopia
Balance - Ethiopia
Uniformity - all top 10 countries
Clean.Cup - all top 10 countries
Sweetness - all top 10 countries
Cupper.Points - Ethiopia
"""

# Can we classify coffee from different countries based on coffee qualities?

# log transform left-skewed data
df_log_transform = pd.DataFrame(df['Country'], columns=['Country'])
for each in columns:
    df_log_transform[each] = np.log(df[each] + 1)

# PCA 
features = columns
X = df_log_transform.iloc[:, 1:].values
Y = df_log_transform.iloc[:, 0].values
# standardize data for mean of 0, variance of 1 
X_std = StandardScaler().fit_transform(X)

'''1. Eigendecomposition'''
# covariance matrix, create eigenvalues and eigenvectors
covariance_mat = np.cov(X_std.T) 
eig_vals, eig_vecs = np.linalg.eig(covariance_mat)

# Covariance matrix
# if positive, two variables increase/decrease together (correlated)
# if negative, two variables go opposite directions (inversely correlated)
print('Covariance matrix')
print(tabulate(covariance_mat))
print('Eigenvectors')
print(tabulate(eig_vecs))
print('Eigenvalues')
print(eig_vals)
# check if eigenvectors are all units of 1
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('All eigenvectors have same length of 1', )

'''2. Selecting Number of Principal Components'''
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# See explained variance for each principal component
total = sum(eig_vals)
ind_variance_list = []
cum_variance = 0
cum_variance_list = []
components = sorted(eig_vals, reverse=True)
df_components = pd.DataFrame(components, columns={'Eigenvalues'})

for each in sorted(eig_vals, reverse=True):
    ind_variance_list.append((each/total)*100)
    cum_variance += each
    cum_variance_list.append(str(round(((cum_variance/total)*100), 2)) + '%')

df_components['Cumulative Variance'] = cum_variance_list
df_components['Individual Variance'] = ind_variance_list
print(tabulate(df_components, headers='keys', tablefmt='psql'))

# Matplotlib table 
fig, ax = plt.subplots(figsize=(7, 7))
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=df_components.to_numpy(),
                 colLabels=['Eigenvalues', 'Cumulative Variance', 
                            'Individual Variance'],
                 loc='center')
table.set_fontsize(10)
#table.scale(1.8, 1.8)
fig.tight_layout()

# 3. PCA projection to 3D
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X)
df_PCA = pd.DataFrame(data=principal_components, columns=
                    ['PC1', 'PC2', 'PC3'])
df_PCA = pd.concat([df_PCA, df_log_transform.iloc[:, 0]], axis =1)

df_importance = pd.DataFrame(pca.components_, columns=features)
df_importance.insert(0, 'Components', ['PC1','PC2', 'PC3'])
print(tabulate(df_importance, headers='keys', tablefmt='psql'))

# Matplotlib table 
fig, ax = plt.subplots(figsize=(10, 9))
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
test = ['Country', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity',
          'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 
          'Cupper.Points']
able = ax.table(cellText=df_importance.round(2).to_numpy(),
                colLabels=test,loc='center')
table.set_fontsize(20)
#table.scale(1.2, 1.2)
fig.tight_layout()

# 3D Visualization
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')
ax.set_xlabel('PC1 (51.93%)', fontsize = 10)
ax.set_ylabel('PC2 (16.11%)', fontsize = 10)
ax.set_zlabel('PC3 (7.11%)', fontsize = 10)
ax.set_title('3 component PCA', fontsize = 15)

xdata = df_PCA['PC1'].to_numpy()
ydata = df_PCA['PC2'].to_numpy()
zdata = df_PCA['PC3'].to_numpy()

targets = top_10
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown',
          'limegreen']
for target, color in zip(targets,colors):
    indicesToKeep = df_PCA['Country'] == target
    ax.scatter3D(df_PCA.loc[indicesToKeep, 'PC1'],
                 df_PCA.loc[indicesToKeep, 'PC2'],
                 df_PCA.loc[indicesToKeep, 'PC3'],
                 label=target,
                 c = color,
                 s = 5)
    
plt.legend(loc=2, prop={'size': 8})
