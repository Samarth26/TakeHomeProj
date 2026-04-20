# Income Prediction (-50K vs 50K+) 

## Data Exploration and Analysis
- The nature of the problem is a binary classification problem where we are trying to predict whether an individual's income exceeds $50K per year based on various features such as age, education, occupation, etc.
- The dataset contains a mix of numerical and categorical features. Some of the key features include:
    - Age: A numerical feature representing the age of the individual.
    - Sex: A categorical feature indicating the gender (M/F).
    - Education: A categorical feature representing the highest level of education attained.
    - Capital gains and losses: Numerical features representing the capital gains and losses of the individual. With Capital Gains capped at 99k
    - Wage per hour: A numerical feature representing the wage per hour, capped at 9.9k.
    - Dividends from stocks: A numerical feature representing the dividends received from stocks.
    - Weeks worked in year: An ordinal feature representing the number of weeks worked in a year.
    - Detailed household and family stat: A categorical feature. 
    - tax filer stat: A categorical feature representing the tax filing status of the individual.
    - Major Industry code, detailed industry recode: Categorical features representing the industry in which the individual works.
    - Detailed occupation recode: A categorical feature representing the occupation of the individual.

- One of the first things to notice is the heavy class imbalance in the target variable, with a ration of 1:15 between -50K:50K+.

- Another crucial observation is that the data is well populated with very few missing values, and '?' which probably represents an answer that was not able to be deciphered during data collection. We can treat these as missing values and impute them with the most frequent value in the respective columns.

- Not in universe is a term used in the dataset to indicate that a particular feature is not applicable to an individual.

- The columns with a large number of Not in universe values which still provide information for prediction are:
    - 'major industry code'
    - 'major occupation code'
    - 'class of worker'

- Columns with a large number of Not in universe were also columns that did not have a lot of importance in the model. However, the large existence of Not in universe values is not the reason for the low importance of these features. 

## Data Cleaning and Preprocessing
- The labels - 50000 and 50000 + were converted to 0 and 1 respectively for easier modeling.
- The missing values represented by '?', 'Not identifiable' and np.nan were replaced with na. 
- We removed the column with more than 50% missing values which was 'migration code-change in msa'.
- The missing values in the remaining columns were a small percentage of the total values and were handled better by our model rather than being imputed, so we left them as is.
- 'Detailed Occupation recode' and 'Detailed Industry recode' were converted from numerical to categorical features and converted to string type. This is to prevent the model from treating them as ordinal features and to allow it to capture the categorical nature of these features.

## Numerical Features
- After our data cleaning and preprocessing steps, we had 6 numerical features: age, capital gains, capital losses, wage per hour, dividends from stocks, and weight. 
- Reminder that the wage per hour and capital gains and dividends from stocks were capped at 9.9k and 99k and 99k respectively. A precaution taken by the census to prevent identification of high income individuals.  
- Weight is a feature that represents the number of people in the population that each row in the dataset represents. It is used to account for the sampling design of the dataset and to ensure that the model's predictions are representative of the overall population.
- We also created a new feature called total income which is the annualised wage of the individual.
    - We annualised the wage by multiplying the wage per hour by 35 which is defined by the Census as the classification for a full-time worker. We then multiply that number by 52. On account of if the individual is a part-time worker, we divide the annualised wage by 2. We find that information in the 'full or part time employment stat' column.
    - An avid reader would question why we have not included capital gains and dividends from stocks in the total income feature. They would also question why then is prediction of income is even a problem given these features. 
        - The issue with the latter is that in the minority class (50K+), those who have responded to have worked 52 weeks in a year, have not necessarily responded to have a high wage per hour. In fact, most of those individuals actually have a wage per hour of 0. 
        - The issue with the former is largely because capital gains itself is a very strong and rather sparse indicator. Including it in the total income feature did not provide much of a boost to the model performance and it also made the feature less interpretable.

- Distribution of numerical features:
![alt text](image-1.png)

## Categorical Features
- Categorical features are the rest of the 35 features in the dataset.
- Majority of these features are nominal categorical features. 
- This makes the 'decision' to use a 'decision tree-based' model easier because of the models ability to split on criterion. This allows for us to also have a non-linear decision boundary which is important given the complexity of the problem.
- There are features with cardinality of 2 for Sex, year to 53 for weeks worked in year.
- Largely the data distribution of the categorical features is skewed with a majority of the data points belonging to a few categories. When the priority is interpretability, we can consider grouping the categories with low frequency into an 'other' category. 
    - An example of this is are categories such as country of birth, hispanic origin, country of birth father, country of birth mother which have a large number of categories with a long tail distribution. 
    - These high-dimensional categorical features add to the reason to swing towards a decision tree-based model as opposed to a linear model which would require one-hot encoding and would not be able to capture the relationships between the categories as effectively.


## Model Selection and Training
- We chose to use XGBoost as our model for this problem due to its ability to handle both numerical and categorical features, its robustness to missing values, and its strong performance on classification tasks.
- We used Optuna for hyperparameter tuning to find the best parameters for our XGBoost. Optuna allows us to efficiently search through the hyperparameter space and provide a range of values instead of a specific set of values, which can lead to better exploration and potentially better performance.
- XGBoost uses a boosting algorithm that builds an ensemble of weak learners (decision trees) to create a strong predictive model.
- It uses L2 regularization in its object function to prevent overfitting. 
- It works by creating estimators sequentially, where each new estimator focuses on correcting the errors made by the previous ones. The final prediction is a weighted sum of the predictions from all the estimators. 
- The hyperparameters we tuned were:
    - max_depth: The maximum depth of the trees. A deeper tree can capture more complex patterns but can also lead to overfitting.
        - Range 
    - max_leaves: The maximum number of leaves in a tree. This can help control the complexity of the model and prevent overfitting.
    - n_estimators: The number of trees in the ensemble. More trees can improve performance but also increase training time.
- We focus on these hyperparameters due to their impact on performance and their ability to control overfitting.
- For the training process, we used 3-fold cross-validation to evaluate the performance of the model on the training data and to prevent overfitting. We also used sample weights to account for the class imbalance in the dataset.
- The loss function used is the binary logistic loss function
    - $ L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] $
    - Where $y_i$ is the true label and $\hat{y}_i$ is the predicted probability for the positive class.

## Evaluation
- We used macro F1 score which is calculated as follows:
    - F1 Score for each class = $ \frac{2 \times (\text{Precision} \times \text{Recall})}{\text{Precision} + \text{Recall}} $
    - Macro F1 Score = $ \frac{\sum_{i=1}^{c} F1_i}{c} $

## Feature Importance and Selection
- Once we built our initial model, we looked at the feature importance scores to understand which features were most influential in predicting the target variable.
- The most important features are the ones that give us a strong signal about those labelled 50K+ (1), those are the ones that belong largely to the 50K+ class and not as prevalent in the -50K class. Given the class imbalance, this is a more useful way to look at feature importance. 
    - Doing so makes it obvious why Detailed Occupation recode performs so well as a feature. The feature that dominates for the 50K+ class is very sparesly represented in the overall dataset and there it's a great discriminator for the 50K+ class.
- There are other features of the sort such as 'major occupation code' however it's important to notice the colinearity between these features and why one outperforms the other.
- The following table shows the collinearity between the 'major occupation code' and 'detailed occupation recode' features and and the better lift that detailed occupation recode provides in terms of the >50K rate.


| filter                                  |   total |   >50K | >50K rate   |
|:----------------------------------------|--------:|-------:|:------------|
| major == Executive admin and managerial |   12495 |   3593 | 28.8%       |
| detailed recode == 2                    |    8756 |   2821 | 32.2%       |
| both conditions (intersection)          |    8756 |   2821 | 32.2%       |

> - Intersection covers **70.1%** of 'Executive admin' rows and **100.0%** of 'detailed recode 2' rows.
> - 'Detailed recode 2' has a higher >50K rate. 33% of coverage in the minority class is a strong signal given that the minority class only makes up about 6% of the overall dataset. 
> -  'Executive admin and managerical covers' has 100% coverage of the detailed recode == 2 and >50K labeled rows, however, it also includes a large number of rows that do not belong to the 50K+ class which is why it performs worse as a feature.

- Bearing that in mind, we can now look at the feature importance scores:

![Feature Importance: ](feature_importance_baseline.png)

- Looking at the feature importance scores, we can now account to why major occupation code scores so low while detailed occupation recode scores so high.

- We take our point further and plot the feature importance scores after having dropped the columns detailed occupation recode and detailed industry recode to show the colinearity between the features and how the importance of major occupation code increases after dropping detailed occupation recode similarly for industry.

- ![Feature Importance after dropping Detailed Occupation & Industry Recode: ](feature_importance_colinearity_check.png)

> Note how Major Occupation and Major Industry code have a significant increase in importance after dropping the detailed occupation and industry recode features. This shows the colinearity between these features and how they are providing similar information to the model.

- We then can remove such columns with high colinearity and low importance and retrain the model to reduce the complexity of the model and to make it more interpretable while preserving model performance. 

                        ── Model Comparison ──
                  model  n_features  train_f1  test_f1    gap
      Baseline (all features)    40    0.8956   0.7621 0.1335
      Selected features          30    0.8811   0.7590 0.1221

- We follow this methodology of looking at feature importance, understanding the features and their relationships with each other and the target variable, and then selecting features based on that understanding to build a more interpretable model while preserving performance.

- We select all the features except the lowest 10 features, i.e. until Major industry code since those are proven to be colinear and all the features after those thereby are either colinear themselves or have very low importance. 

## Model Usage Recommendation 
- Any data collected in the future should be preprocessed in the same way as the training data, including handling missing values, creating the total income feature, and ensuring consistency in the labels of the categorical features.
- If the priority is targeting all the 50k+ individuals, then the model can be used with a lower threshold to increase recall at the cost of precision. We could provide that as an argument --prediction_threshold to the predict function in the pipeline.py script. Where the default value is 0.5, and higher values increase the threshold for classifying an individual as 50K+ and lower values decrease the threshold.
- Further, the model's out-of-distribution performance should be monitored closely given the class imbalance especially if using the Baseline Model.
- Lastly, a consideration for feature selection would be any data quality knowledge and costs, if there are features that are costly to collect or known to have data quality issues which are not visible to us, then it could be a good idea to drop those features and find a replacement. Such as using 'major occupation code' instead of 'detailed occupation recode'.

# Segmentation Model 

## Data Exploration and Analysis
- Largely a segmentation problem starts with being able to represent the data properly. Second it is to use that representation to be able to segment the data in a way that is useful for the business problem at hand.
- For the first part we select features that would be useful for the marketing team and that would have a meaningful representation. 
    - For example, some columns could be 90% one label are unlikely to bring much value to the segmentation model. Second some columns which are of high cardinality and have a long tail distribution are also unlikely to bring much value to the segmentation model.
    - Therefore we select the following features for the segmentation model:
        - numerical_cols = ['age', 'capital gains', 'capital losses', 'dividends from stocks', 'weeks worked in year', 'wage per hour']
        - categorical_cols = ['major occupation code', 'major industry code', 'education', 'detailed household summary in household', 'detailed household and family stat', 'tax filer stat', 'full or part time employment stat', 'race', 'marital stat', 'veterans benefits', 'country of birth self', 'citizenship' ]
        - These features are selected based on their importance to the marketing team and for the categoircal columns it is also based on their distribution and cardinality.

## Model Selection and (Training (not really training but more of a clustering technique selection))
- For the second part, we use a dimensionality reduction technique such as PCA or t-SNE to reduce the dimensionality of the data and hypothesise which clustering technique would be best suited for the data. 
- For this one can use PCA or t-SNE, both provided great visuals but t-SNE in particular provided a very clear visual of the clusters in the data.
![Tsne:](tsne_raw.png)
- We then look at the dimension at which we capture 90% of the variance in the data and use that as the number of dimensions to reduce the data to. This is because PCA tries to find eigenvectors that capture the most variance in the data and by looking at the explained variance ratio, we can find the number of dimensions that capture a significant amount of variance in the data. Eigenvectors are orthogonal vectors and the ones on the covaraince matrix capture the directions of maximum variance in the data. 
![PCA Explained Variance: ](pca_explained_variance.png)
- After reducing the dimensionality of the data, we apply Elbow Plot and Silhouette Plot to get the number of clusters we can use. Where Elbow plot uses within sum of squares to tell us how far each point is from its cluster centroid and the silhouette plot gives us how well separated the clusters are and the tightness.
![Elbow and Silhouette Plot:](kmeans_elbow_silhouette_nb.png)
- The plots pointed at using K-means with 6 clusters as a good clustering technique for our data.
- Once we plot the clusters, we can then look at the distribution of the features in each cluster to be able to provide insights on the characteristics of each cluster and how they differ from each other.

![Clustered Tsne: ](tsne_clusters.png)

- The clusters can be nicely defined as follows:
    - Cluster 1 and 0: Younger and Older individuals respectively with small capital gains and losses. Larger dividends from stocks for the older generation and non-existent for the younger generation, and basically little to no wage per hour. They are likely to be students and retirees. There are other obvious differences in the distribution of the categorical features such as education, marital status, etc. Which adds to the well defined nature of these clusters.
    - Cluster 2 to 5: These Clusters are middle aged individuals. Grouped based on different similarities. 
        - Cluster 2 and 5: These are working individuals whose primary source of income is wage per hour. They have small to medium capital gains and losses and dividends from stocks. Two bigger differences between the two is their age with the latter being older and also more likely to be a householder.
        - Cluster 3 and 4: Both of similar age but the former has a much higher capital gains and dividends from stocks while the latter has a much higher wage per hour a good amount of dividends from stocks but also much higher capital losses. They both have large appetites for risk. 

- We can attach back our target variable to the clusters. It then becomes obvious that cluster 3 is the cluster with the highest percentage of 50K+ and individuals in the cluseter 0 and 1 are the least likely to be 50K+. What is less obvious is why our cluster 2 and 5 have quite a low percentage of 50K+ individuals despite being working individuals. This exercise also does show us why Capital Gains was such an important feature for predicting 50K+. 



References:
- https://xgboost.readthedocs.io/en/stable/parameter.html#cat-param
- https://kdd.ics.uci.edu/databases/census-income/census-income.names
- https://arxiv.org/pdf/1603.02754
- Claude Code was used to consolidate the findings from clean_Obj_1.ipynb to the script pipeline.py and segementation.ipnyb to segmentation.py. 