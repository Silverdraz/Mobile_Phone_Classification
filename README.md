# mobile Price Classification (Multi-class Classification Machine Learning)
**The objective of this ML github project is to showcase the end-to-end pipeline of multiclass machine learning classification problem from data ingestion to data preprocessing to feature engineering to model comparison to hyperparameter tuning and eventually inferencing/predicting.

## Scoring Metric Choice
**Accuracy is used as the choice of evaluation metric** as from EDA, it can be observed that the **price_range target labels are well balanced.** The model evaluated using accuracy should generalise well to test dataset, provided that the distribution of the test dataset is similar to the train dataset, though this is regardless of the model. 

## Exploratory Data Analysis + Data Preprocessing + Modelling Approaches 
**Exploratory Data Analysis** is performed to gain a better understanding of the data so that **better insights and decision-making, such as removal of columns, understanding of columns, feature engineering of new, and checking of missing values percentage can all be performed.**

1. **Data Preprocessing**
a.   **Remove redundant features such as “Unnamed: 0” since from domain knowledge**, unique identifiers do not provide signals or aid in the predictive performance of the machine learning models.

b. **Creation of new variables (Feature Engineering) – “old” feature.** First, using domain knowledge, a **set of features that new phones will possess are – wifi, Bluetooth, dual_sim, 4G and touch_screen.** As such, phones that do not possess any one of these features are considered as “old” phone model. It is not required to create a “new” feature as it will be a linear combination of the “old” feature, which will increase the number of feature, while having a high redundancy of providing the same information as the new (new = 1 – old)

c. **Creation of new variables to model non-linearity (spline transformation).** From EDA, it is observed that price_range has a non-linear distribution across “Ram” variable. **Furthermore, the distributions across price_range greatly differ among each categories across the ram feature, allowing ml models to better classify and separate the price_range categories.** Histograms/Binning was used to check for non-linearity, but it is not advisable as binning leads to a loss of information and the cut-offs are arbitrary. Instead, **spline** was used to model non-linearity.

d. **Missing Values were handled by multiple imputation** as introducing mean or mode imputation can distort the model distribution, and it is not best practice. However, it is done in fold to prevent data leakage or over-optimistic results from being reported, contributing to a slightly longer latency. Complete-case analysis was not chosen as it will remove rows completely, reducing the number of samples used for training.

e. **Multicollinearity check was not performed as the primary objective of the model and task is to best predict the price_range.**  Perfect multicollinearity would however be an issue as it introduces variables without providing more signals or information to the model. As such, the “new” phone model variable was not introduced together with the “old” phone models variable. Otherwise, the original features in the excel sheet do not seem to be a linear combination of each other.

2.  **Modelling & Model Comparison**
a.	**Cross validation with shuffling was used** to prevent machine learning models from learning any ordering. **Random seed was used for consistency** and comparison, otherwise the performance will fluctuate.

b.	**XGBoost Classifier was used as a baseline as it provides a good performance without any tuning and can handle missing values without imputation.** A high bar was thought to be better as otherwise it may lead to overtly optimistic results when the baseline model used has lower performance benchmark.

c.	**Iterative improvements were added to the baseline model to check if the model performance can be improved.**
i.	Adding new features together with multiple imputation  had higher score than the baseline, suggesting that they do indeed provide better signals to the model and improve predictive performance
ii.	Spline transformation reduced validation accuracy in both the baseline (XGBoost) and final selected model SVM. It was hence not introduced to the final model fitting
iii.	Comparison across the models were then performed from non-tree base models to tree-based models. **Surprisingly, non-tree based models, such as KNN and SVM radial performed better than the trees based models.** SVM was selected as the final fitted model since it has the highest validation accuracy with the least overfitting as shown from the mean train acc and mean val accuracy.
iv.	**SVM performed best with the feature engineered “old” phone model variable**, compared to using the original datasets or the spline transformation. **Standardized (standard scalar) has been trialled as SVM uses distance metrics to classify and having a standardized feature set after imputation and before the model should be better.** Surprisingly, the model works poorer under these conditions and was as such, these were not adopted.
v.	Grid search hyperparameter search was implemented as it was thought there is slight overfitting from the results.csv file. However, the param of C = 1.1 provided the best score, which indicates that lower regularisation instead provided better validation accuracy. Hence, perhaps the overfitting is considered insignificant


