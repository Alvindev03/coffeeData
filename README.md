# Coffee Quality Analysis
Welcome to my Coffee Quality Analysis project!

The dataset for this project was taken from [jldbc](https://github.com/jldbc/coffee-quality-database). I wanted to practice my data analysis skills and decided to use this particular dataset to understand if coffee reviews from different parts of the world differ.

## Data
The data was collected in 2018 and I focused on these particular measures of quality (features):

* Aroma
* Flavor
* Aftertaste
* Acidity
* Body
* Balance
* Uniformity
* Cup Cleanliness
* Sweetness

## Questions

1. In this dataset, which countries produce the most coffee?

2. Are the average total ratings of coffee from the top 10 country similar?

3. Do we need all category ratings to describe the coffee?

  

## Methods

### Q1: 
* Ranked bar chart and boxplots

### Q2
* Shapiro-Wilks and D'Agostino-Pearson: test normality
* Kruskal-Wallis: Non parametric test for similarity between groups
* Dunn Test: identify different groups

###  Q3
* PCA: Reduce dimensions and visualizaition

## Results

### Q1
Top 10 coffee suppliers are:
1. Mexico
2. Colombia
3. Guatemala
4. Brazil
5. Taiwan
6. United States (Hawaii)
7. Honduras
8. Costa Rica
9. Ethiopia
10. Tanzania

![Frequency bar chart](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Figure_1.png?raw=true)

 

![Box plot](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Figure_2.png?raw=true)

### Q2

The majority of each group (countries) had an non-Gaussian distribution.

Kruskal-Wallis test determined there were differences in overall coffee quality and Dunn Test identified some groups to be different (see image).

  
![Dunn Test](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Figure_3.png?raw=true)


### Q3

When examining covariance, three principal components captured 82.6% of explained variance.

  
![Variance matrix](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Figure_4.png?raw=true)

 
Coffee from all countries clustered together with Mexico (red dots) least amount of clustering.

![PCA](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Figure_6.png)

## Discussion
What did I learn?
Coffee from around the world are ranked very similarly or that I need to expand my dataset to better discriminate coffee from different countries. 

## Future Directions
Will webscrape 2019 (most recent) dataset and use more features.