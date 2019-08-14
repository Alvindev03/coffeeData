import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

with open('arabica_data_cleaned.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
    print(f'Processed {line_count} lines.')

df = pd.read_csv('arabica_data_cleaned.csv')
df1 = df[['Country.of.Origin', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity',
          'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness']]
df1['Country.of.Origin'].replace('Tanzania, United Republic Of', 'Tanzania',
                                  inplace=True)
df1 = df1.rename(columns={'Country.of.Origin': 'Country'})
country_list = df1['Country'].unique().tolist()
country_list.pop(-2) # removes 'nan' country

df_histogram = pd.DataFrame(df1['Country'].value_counts())
df_histogram.reset_index(level=0, inplace=True)
df_histogram.columns=['Country', 'Number of Suppliers']

"""
1. Which country has the most coffee suppliers tested?
* Mexico

Since ~82% of coffee supplies are from the top 10 countries, I will be focusing 
on them for the remainder of analysis
print(df_histogram['Number of Suppliers'][0:10].sum()/
      df_histogram['Number of Suppliers'].sum())
"""
# frequency histogram
x = df_histogram['Country']
y = df_histogram['Number of Suppliers']

fig, ax1 = plt.subplots(figsize=(10, 5), tight_layout=True) 
# figsize = (length, width)
# tight_layout fits graph and labels in window
ax1.bar(x, y, width=0.8)
ax1.set_ylabel('Number of Suppliers')
ax1.set_xlabel('Countries')
ax1.set_title('Coffee Suppliers Around the World')
plt.xticks(x, df_histogram['Country'], rotation=90, fontsize=10)

"""
2. What is the average total rating of each country's coffee? 
# out of 90 total points
"""
df1['Total'] = df1.sum(axis=1)
top10_countries = df_histogram['Country'].tolist()[:10]

country_total = {}
for each in top10_countries:
    country_total[each] = df1[df1['Country']==each]['Total']

boxplot_data = []
for each in country_total:
    boxplot_data.append(country_total[each])

# top subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), 
                               gridspec_kw={'height_ratios': [4, 1]},
                               tight_layout=True, sharex=True)
ax1.set_title('Average International Coffee Supplier Rating (Top 10)')
ax1.boxplot(boxplot_data,)
ax1.set_ylabel('Average Rating (out of 90)')
ax1.set_ylim(bottom=50)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)

# bottom subplot
ax2.spines['top'].set_visible(False)
ax1.boxplot(boxplot_data,)
ax2.set_ylim(top=20)
ax2.set_xticklabels(labels=top10_countries)
ax2.set_xlabel('Countries')
plt.xticks(rotation=-60, fontsize=10)

plt.show()
"""
3. Is there a difference in quality between each country's coffee?

3a. Check data for normality 
* Shapiro-Wilk test for Gaussian distribution
* D'Agostino-Pearson test for skewedness and kurtosis

Only 3 of the 10 countries fail to reject Shapiro-Wilks H0 (Gaussian dist.).
- Taiwan, Ethiopia, Tanzania 

Only 2 of the 10 countries fail to reject D'Agostino-Pearson H0 (Gaussian dist.)
- Taiwan, Ethiopia
"""
shapiro = {} # for Shapiro-Wilks
normaltest = {} # for D'Agostino-Pearson

for each in top10_countries:
    shapiro[each] = stats.shapiro(country_total[each].tolist())
    normaltest[each] = stats.normaltest(country_total[each].tolist())

    print(each)
    shapiro_calc_p = shapiro[each][1]
    if shapiro_calc_p > 0.05:
        print(round(shapiro_calc_p, 3),
              'Gaussian dist. - fail to reject Shapiro-Wilks H0')
    else:
        print(round(shapiro_calc_p, 3),
              'Non-Gaussian dist. - reject Shapiro-Wilks H0')
    
    normaltest_calc_p = normaltest[each][1]
    if normaltest_calc_p > 0.05:
        print(round(normaltest_calc_p, 3), 
                    'Gaussian dist. - fail to reject D\'Agostino-Pearson H0')
    else:
        print(round(normaltest_calc_p, 3),
                    'Non-Gaussian dist. - reject D\'Agostino-Pearson H0')

    print()
"""
3b. One-way ANOVA or Kruskall-Wallis test
* Kruskall-Wallis nonparametric test b/c data is abnormal
"""
Mexico_samples = country_total['Mexico'].tolist()
Colombia_samples = country_total['Colombia'].tolist()
Guatemala_samples = country_total['Guatemala'].tolist()
Brazil_samples = country_total['Brazil'].tolist()
Taiwan_samples = country_total['Taiwan'].tolist()
United_States_Hawaii_samples = country_total['United States (Hawaii)'].tolist()
Honduras_samples = country_total['Honduras'].tolist()
Costa_Rica_samples = country_total['Costa Rica'].tolist()
Ethiopia_samples = country_total['Ethiopia'].tolist()
Tanzania_samples = country_total['Tanzania'].tolist()

kruskal_calc_p = stats.kruskal(Mexico_samples, Colombia_samples,
                               Guatemala_samples, Brazil_samples,
                               Taiwan_samples, United_States_Hawaii_samples,
                               Honduras_samples, Costa_Rica_samples,
                               Ethiopia_samples, Tanzania_samples)[1]

print('Kruskall-Wallis Result: ', kruskal_calc_p)
if kruskal_calc_p > 0.05:
    print('Not significant: fail to reject H0')
else:
    print('Significant: reject H0')
"""
4. Determine which countries have vastly different ratings than others 
* Dunn Test
+----+----------+---------------+---------------+---------------+---------------+--------------------------+---------------+---------------+---------------+--------------+
|    |   Mexico |      Colombia |     Guatemala |        Brazil |        Taiwan |   United States (Hawaii) |      Honduras |    Costa Rica |      Ethiopia |     Tanzania |
|----+----------+---------------+---------------+---------------+---------------+--------------------------+---------------+---------------+---------------+--------------|
|  1 |       -1 |   1.68477e-25 |   9.88559e-07 |   7.41973e-07 |   0.0165658   |              0.00172229  |   0.705141    |   2.10412e-07 |   7.3022e-27  |  0.00952734  |
|  2 |      nan |  -1           |   2.06118e-07 |   1.78016e-05 |   2.20476e-07 |              1.11432e-05 |   4.92514e-10 |   0.152449    |   1.23082e-05 |  0.000808302 |
|  3 |      nan | nan           |  -1           |   0.634122    |   0.226993    |              0.645582    |   0.00637763  |   0.0449474   |   2.81406e-14 |  0.817839    |
|  4 |      nan | nan           | nan           |  -1           |   0.127487    |              0.417564    |   0.00312888  |   0.110127    |   2.03956e-12 |  0.59974     |
|  5 |      nan | nan           | nan           | nan           |  -1           |              0.534434    |   0.147195    |   0.00768619  |   2.81538e-14 |  0.520971    |
|  6 |      nan | nan           | nan           | nan           | nan           |             -1           |   0.0447111   |   0.036509    |   2.01967e-12 |  0.904812    |
|  7 |      nan | nan           | nan           | nan           | nan           |            nan           |  -1           |   0.000149162 |   6.39025e-17 |  0.0654915   |
|  8 |      nan | nan           | nan           | nan           | nan           |            nan           | nan           |  -1           |   3.02717e-06 |  0.0899595   |
|  9 |      nan | nan           | nan           | nan           | nan           |            nan           | nan           | nan           |  -1           |  1.57413e-09 |
| 10 |      nan | nan           | nan           | nan           | nan           |            nan           | nan           | nan           | nan           | -1           |
+----+----------+---------------+---------------+---------------+---------------+--------------------------+---------------+---------------+---------------+--------------+
"""
df_dunn = sp.posthoc_dunn([Mexico_samples, Colombia_samples,
                               Guatemala_samples, Brazil_samples,
                               Taiwan_samples, United_States_Hawaii_samples,
                               Honduras_samples, Costa_Rica_samples,
                               Ethiopia_samples, Tanzania_samples])

df_dunn.columns = top10_countries
np.triu(np.ones(df_dunn.shape)).astype(np.bool)
df_dunn = df_dunn.where(np.triu(np.ones(df_dunn.shape)).astype(np.bool))
df_dunn = df_dunn.round(6)
print(tabulate(df_dunn, headers='keys', tablefmt='psql'))

# Matplotlib table
fig, ax = plt.subplots(figsize=(18, 8))
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=df_dunn.to_numpy(), colLabels=top10_countries,
                 loc='center')
table.set_fontsize(30)
#table.scale(1.8, 1.8)
fig.tight_layout()

"""
5. Determine which features are important
PCA - dimension reduction 
Remember: PCA maximizes variance
+----+---------------+-----------------------+-----------------------+
|    |   Eigenvalues | Cumulative Variance   |   Individual Variance |
|----+---------------+-----------------------+-----------------------|
|  0 |     5.54137   | 61.52%                |              61.5238  |
|  1 |     1.40866   | 77.16%                |              15.6398  |
|  2 |     0.491423  | 82.62%                |               5.45609 |
|  3 |     0.463196  | 87.76%                |               5.14269 |
|  4 |     0.325438  | 91.38%                |               3.61322 |
|  5 |     0.263463  | 94.3%                 |               2.92514 |
|  6 |     0.236257  | 96.92%                |               2.62308 |
|  7 |     0.178871  | 98.91%                |               1.98594 |
|  8 |     0.0981904 | 100.0%                |               1.09017 |
+----+---------------+-----------------------+-----------------------+
"""
features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance',
            'Uniformity', 'Clean.Cup', 'Sweetness']

X = df1.iloc[:, 1:10].values
Y = df1.iloc[:, 0].values
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
df_PCA = pd.concat([df_PCA, df1.iloc[:, 0]], axis =1)

df_importance = pd.DataFrame(pca.components_, columns=features)
df_importance.insert(0, 'Components', ['PC1','PC2', 'PC3'])
print(tabulate(df_importance, headers='keys', tablefmt='psql'))

# Matplotlib table 
fig, ax = plt.subplots(figsize=(7, 7))
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

test = ['Components', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
        'Balance','Uniformity', 'Clean.Cup', 'Sweetness']
able = ax.table(cellText=df_importance.round(2).to_numpy(),
                colLabels=test,loc='center')
table.set_fontsize(80)
#table.scale(1.2, 1.2)
fig.tight_layout()

# 3D Visualization
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')
ax.set_xlabel('PC1 (61.52%)', fontsize = 10)
ax.set_ylabel('PC2 (15.64%)', fontsize = 10)
ax.set_zlabel('PC3 (5.46%)', fontsize = 10)
ax.set_title('3 component PCA', fontsize = 15)

xdata = df_PCA['PC1'].to_numpy()
ydata = df_PCA['PC2'].to_numpy()
zdata = df_PCA['PC3'].to_numpy()

targets = top10_countries
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown',
          'limegreen']
for target, color in zip(targets,colors):
    indicesToKeep = df_PCA['Country'] == target
    ax.scatter3D(df_PCA.loc[indicesToKeep, 'PC1'],
                 df_PCA.loc[indicesToKeep, 'PC2'],
                 df_PCA.loc[indicesToKeep, 'PC3'],
                 label=target,
                 c = color,
                 s = 15)
plt.legend(loc=2, prop={'size': 8})
plt.show()

