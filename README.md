# Coffee Quality Analysis
Welcome to my Coffee Quality Analysis project!

## Motivation
This a personal project to help practice my Python data analysis skills. While I've drank coffee for many years, I've never explored where coffee comes from and the resulting differences in taste. I figured playing with this dataset will help give me insight on understanding the different types of coffee qualities and hopefully any differences based on its country of origin. 

## Summary
Coffee Quality Institute coffee quality data was analyzed to determine the top countries with the most coffee review and country with the highest quality coffee.
Mexico was found to be the most reviewed country with a total of 236 reviews and Ethiopia was determined to have the highest quality coffee. A principal component analysis was also conducted, but data failed to capture enough variance to discern between coffee belonging to different countries. 

## Data
The dataset for this project was taken from Github user: [jldbc](https://github.com/jldbc/coffee-quality-database), who collected this in 2018. Each quality rating has a score from 0 to 10 with the maximum total points of 100 and has been rated by trained coffee tasters. 

For my project, I focused on these particular measures of coffee quality:
* Aroma
* Flavor
* Aftertaste
* Acidity
* Body
* Balance
* Uniformity
* Cup Cleanliness
* Sweetness
* Cupper Points

Note: I am only working with arabica (vs. robusta) since it has a larger dataset to work with. 

## Questions

1. Which countries have most amount of coffee ratings?

2. Of the top 10 countries, which country has best quality coffee?

3. Can we classify coffee from different countries based on coffee quality ratings?
  
## Methods

### Data Exploration
    1. Pearson Correlation Matrix
    2. Boxplots
    3. Kernel Density Estimation Distribution

### Q1: 
* Ranked bar chart

### Q2
* Boxplots for each country
* Median value taken instead of mean due too outliers

###  Q3
* PCA: Reduce dimensions and visualization

## Results

### Data Exploration

![Heatmap](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Pearson_Heatmap.png?raw=true)

![Box plot](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Overall_Coffee_Quality.png?raw=true)

![Density distribution](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Overall_Coffee_Quality_Dist.png?raw=true)

### Q1
Top 10 coffee suppliers are: 
(Country, Number of Reviews)
1. Mexico                  236
2. Colombia                183
3. Guatemala               181
4. Brazil                  132
5. Taiwan                   75
6. United States (Hawaii)   73
7. Honduras                 53
8. Costa Rica               51
9. Ethiopia                 44
10. Tanzania                40

![Frequency bar chart](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Coffee_Suppliers.png?raw=true)

### Q2
Because there were many outliers within dataset and I did not want to trim since they are valid in this dataset, I decided to look for the country with the highest
median value in each quality category. 

Ethiopia's coffee had the highest ratings for each category except for uniformity, clean cup, and sweetness where all countries shared the same median value of 10.

![Ethiopia](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/Ethiopia_Coffee_Quality.png?raw=true)

Please see the images folder for coffee quality ratings of other countries.

### Q3
When examining covariance, three principal components captured 75.15% of explained variance.
![Cumulative variance](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/PC_Variance.png?raw=true)

There were no distinct grouping seen in the PCA
![PCA](https://github.com/timmy224/Coffee_Quality_Analysis/blob/master/images/PCA.png?raw=true)

## Discussion
After analyzing this dataset, Mexico has the most number of coffee quality reviews with a total of 236 reviews. When examining coffee qualities, Ethiopia seems to have the highest quality of coffee based on 44 coffee quality reviews. When trying to determine whether distinct groups of coffee from different countries can be determined from coffee quality ratings, the PCA failed to capture enough variance from the given data to suggest conclusions to do so. 

## Future Directions
Because other parts (non-quality measures) of the dataset were vastly inconsistent/incomplete, I omitted incorporating them into my analysis. I may consider incorporating them to help provide extra data for the PCA to work with to hopefully capture more variance. 

