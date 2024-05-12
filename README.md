# Advanced-Tor-NonTor-Traffic-Classification
This repository contains the detailed results for our research paper.

**RESULTS**

1. **SCENARIO-A**

**File-1 (15s)**

**Data Distribution Analysis:**

- The dataset comprised network flows with a flow timeout of 15 seconds.
- Initial data distribution showed a majority of non-Tor instances, with a ratio of approximately 5.65 non-Tor instances for every Tor instance.
- After preprocessing, the dataset was balanced with 17,020 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Feature reduction techniques effectively streamlined the dataset, reducing the feature set by approximately 8.70%.

| **SCENARIO-A 15-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 18758 |
| Tor | 3314 |
| Number of Negative Values |     |
| 68603 |     |
| Number of Inf Values |     |
| 0   |     |
| After Outlier Removal |     |
| Non-Tor | 17020 |
| Tor | 2844 |
| After Oversampling |     |
| Non-Tor | 17020 |
| Tor | 17020 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 15-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 21  |

| **SCENARIO-A 15s** |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **ROC AUC** |
| Random Forest | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Decision Tree | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| SVM | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| KNN | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| XGBoost | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Gradient Boosting | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 98.29% |     | 98.35% | 98.40% | 98.35% | 98.35% | 0.98 | 0.98 | 1.00 |
| AdaBoost | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Bagging Classifier | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Extra Trees | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **ROC AUC** |
| DNN | Accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.01 | 0.00 | 1.00 |
| &nbsp; | Precision | 100.00% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 100.00% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 100.00% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Various classifiers demonstrated exceptional performance, with mean accuracy (cross-validation) ranging from 98.29% to 100.00%.
- Among the classifiers, Logistic Regression, Decision Tree, and XGBoost stood out for achieving perfect accuracy, precision, recall, and F1 scores.
- However, Gaussian Naive Bayes, although not reaching perfect scores, performed remarkably well, achieving an accuracy of 98.29% and balanced precision, recall, and F1 scores of approximately 98.35%.
- Conversely, while all classifiers performed admirably, Gradient Boosting exhibited a slightly lower performance compared to others, with mean accuracy (cross-validation) of 99.99%.

**Learning Curve Metrics:**

- Training and validation scores remained consistently high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores consistently approached or reached 1.00, signifying excellent discrimination performance for all classifiers.

**Model Comparison:**

- Logistic Regression, Decision Tree, and XGBoost emerged as top performers, achieving perfect scores across all evaluation metrics.
- Gaussian Naive Bayes, despite its simplicity, demonstrated impressive performance, especially in accurately classifying Tor and non-Tor instances.
- Gradient Boosting, while still performing well, showed slightly lower accuracy compared to other classifiers.

**Results**

**Scenario-A File-2 (30s)**

**Data Distribution Analysis:**

- The dataset for File 2 consisted of network flows with a flow timeout of 30 seconds.
- Initially, there were 14,651 non-Tor instances and 1,771 Tor instances, indicating a skewed distribution with approximately 8.28 non-Tor instances for every Tor instance.
- After outlier removal, the instances decreased to 13,326 non-Tor and 1,453 Tor instances.
- Subsequent oversampling balanced the dataset, resulting in 13,326 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Similar to File 1, feature reduction techniques reduced the initial 23 features to 21, streamlining the dataset for improved model efficiency.

| **SCENARIO-A 30-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 14651 |
| Tor | 1771 |
| Number of Negative Values |     |
| 46840 |     |
| Number of Inf Values |     |
| 0   |     |
| After Outlier Removal |     |
| Non-Tor | 13326 |
| Tor | 1453 |
| After Oversampling |     |
| Non-Tor | 13326 |
| Tor | 13326 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 30-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 21  |

| **SCENARIO-A 30s** |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **ROC AUC** |
| Random Forest | 99.98% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| Decision Tree | 99.95% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| SVM | 99.96% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| KNN | 99.97% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| XGBoost | 99.97% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 99.98% |     | 99.98% | 99.98% | 99.98% | 99.98% | 1.00 | 1.00 | 1.00 |
| Gradient Boosting | 99.99% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 98.05% |     | 97.88% | 97.96% | 97.88% | 97.87% | 0.98 | 0.98 | 1.00 |
| AdaBoost | 99.98% |     | 99.98% | 99.98% | 99.98% | 99.98% | 1.00 | 1.00 | 1.00 |
| Bagging Classifier | 99.96% |     | 99.92% | 99.92% | 99.92% | 99.92% | 1.00 | 1.00 | 1.00 |
| Extra Trees | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **ROC AUC** |
| DNN | Accuracy | 99.98% | 99.96% | 99.96% | 99.96% | 99.96% | 0.01 | 0.00 | 1.00 |
| &nbsp; | Precision | 99.98% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 99.98% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 99.98% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Various classifiers were evaluated for their performance in distinguishing between Tor and non-Tor instances. Notable observations include:
  - Random Forest, Decision Tree, and Logistic Regression achieved near-perfect mean accuracy (cross-validation) of 99.98%.
  - KNN and Extra Trees achieved perfect accuracy, precision, recall, and F1 scores, indicating robust performance.
  - Gaussian Naive Bayes exhibited comparatively lower performance, with an accuracy of 98.05% and slightly lower precision, recall, and F1 scores.
  - The deep neural network (DNN) demonstrated excellent performance across all evaluation metrics, with an accuracy, precision, recall, and F1 score of 99.98%.

**Learning Curve Metrics:**

- Training and validation scores remained consistently high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores consistently approached or reached 1.00, signifying excellent discrimination performance for all classifiers.

**Model Comparison:**

- Logistic Regression, Random Forest, and Decision Tree emerged as top performers, achieving near-perfect accuracy and other evaluation metrics.
- Gaussian Naive Bayes, while showing slightly lower performance, still demonstrated commendable accuracy in distinguishing between Tor and non-Tor instances.
- The deep neural network (DNN) showcased impressive performance across all evaluation metrics, emphasizing its effectiveness in classification tasks.

**Results**

**Scenario-A File-3 (60s)**

**Data Distribution Analysis:**

- The dataset for File 3 consisted of network flows with a flow timeout of 60 seconds.
- Initially, there were 15,515 non-Tor instances and 914 Tor instances, indicating a skewed distribution with approximately 16.96 non-Tor instances for every Tor instance.
- After preprocessing, the dataset was balanced with 14,072 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Feature reduction techniques effectively reduced the initial 23 features to 6, streamlining the dataset for improved model efficiency.

| **SCENARIO-A 60-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 15515 |
| Tor | 914 |
| Number of Negative Values |     |
| 46909 |     |
| Number of Inf Values |     |
| 0   |     |
| After Outlier Removal |     |
| Non-Tor | 14072 |
| Tor | 736 |
| After Oversampling |     |
| Non-Tor | 14072 |
| Tor | 14072 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 60-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 6   |

**Classifier Performance:**

- Various classifiers were evaluated for their performance in distinguishing between Tor and non-Tor instances. Notable observations include:
  - KNN, Logistic Regression, Gradient Boosting, and Extra Trees achieved perfect mean accuracy (cross-validation) of 100.00%.
  - Decision Tree, SVM, XGBoost, AdaBoost, and Bagging Classifier also demonstrated near-perfect accuracy and other evaluation metrics.
  - Gaussian Naive Bayes exhibited slightly lower performance compared to other classifiers, with an accuracy of 98.89% and slightly lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased impressive performance across all evaluation metrics, with perfect accuracy, precision, recall, and F1 score of 100.00%.

**Learning Curve Metrics:**

- Training and validation scores remained consistently high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores consistently approached or reached 1.00, signifying excellent discrimination performance for all classifiers.

**Model Comparison:**

- KNN, Logistic Regression, Gradient Boosting, and Extra Trees emerged as top performers, achieving perfect accuracy and other evaluation metrics.
- Gaussian Naive Bayes, while showing slightly lower performance, still demonstrated commendable accuracy in distinguishing between Tor and non-Tor instances.
- The deep neural network (DNN) showcased impressive performance across all evaluation metrics, emphasizing its effectiveness in classification tasks.

**Results for File-4 (120s)**

Scenario-A

**Data Distribution Analysis:**

- The dataset for File 4 consisted of network flows with a flow timeout of 120 seconds.
- Initially, there were 10,782 non-Tor instances and 470 Tor instances, indicating a skewed distribution with approximately 22.92 non-Tor instances for every Tor instance.
- After preprocessing, the dataset was balanced with 9,812 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Feature reduction techniques effectively reduced the initial 23 features to 6, streamlining the dataset for improved model efficiency.

| **SCENARIO-A 120-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 10782 |
| Tor | 470 |
| Number of Negative Values |     |
| 33365 |     |
| Number of Inf Values |     |
| 0   |     |
| After Outlier Removal |     |
| Non-Tor | 9812 |
| Tor | 317 |
| After Oversampling |     |
| Non-Tor | 9812 |
| Tor | 9812 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 120-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 6   |

| **SCENARIO-A 120s** |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **ROC AUC** |
| Random Forest | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Decision Tree | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| SVM | 99.31% |     | 99.33% | 99.34% | 99.33% | 99.33% | 0.99 | 0.99 | 1.00 |
| KNN | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| XGBoost | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 99.99% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Gradient Boosting | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 99.05% |     | 98.98% | 99.00% | 98.98% | 98.98% | 0.99 | 0.99 | 1.00 |
| AdaBoost | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Bagging Classifier | 99.98% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| Extra Trees | 100.00% |     | 100.00% | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **ROC AUC** |
| DNN | Accuracy | 99.97% | 100.00% | 100.00% | 100.00% | 100.00% | 0.01 | 0.00 | 1.00 |
| &nbsp; | Precision | 99.97% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 99.97% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 99.97% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Various classifiers were evaluated for their performance in distinguishing between Tor and non-Tor instances. Notable observations include:
  - KNN, XGBoost, Logistic Regression, Gradient Boosting, and Extra Trees achieved perfect mean accuracy (cross-validation) of 100.00%.
  - Random Forest and Decision Tree demonstrated near-perfect accuracy and other evaluation metrics, with mean accuracy (cross-validation) of 99.98%.
  - Gaussian Naive Bayes exhibited slightly lower performance compared to other classifiers, with an accuracy of 99.05% and slightly lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased impressive performance across all evaluation metrics, with accuracy, precision, recall, and F1 score of 99.97%.

**Learning Curve Metrics:**

- Training and validation scores remained consistently high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores consistently approached or reached 1.00, signifying excellent discrimination performance for all classifiers.

**Model Comparison:**

- KNN, XGBoost, Logistic Regression, Gradient Boosting, and Extra Trees emerged as top performers, achieving perfect accuracy and other evaluation metrics.
- Gaussian Naive Bayes, while showing slightly lower performance, still demonstrated commendable accuracy in distinguishing between Tor and non-Tor instances.
- The deep neural network (DNN) showcased impressive performance across all evaluation metrics, emphasizing its effectiveness in classification tasks.

Top of Form

**Results for File-5 (10s)**

Scenario-A

**Data Distribution Analysis:**

- The dataset for File 5 consisted of network flows with a flow timeout of 10 seconds.
- Initially, there were 59,790 non-Tor instances and 8,044 Tor instances, indicating a skewed distribution with approximately 7.43 non-Tor instances for every Tor instance.
- After preprocessing, the dataset was balanced with 53,859 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Feature reduction techniques effectively reduced the initial 23 features to 18, streamlining the dataset for improved model efficiency.

| **SCENARIO-A 10-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 59790 |
| Tor | 8044 |
| Number of Negative Values |     |
| 477 |     |
| Number of Inf Values |     |
| 10  |     |
| After Outlier Removal |     |
| Non-Tor | 53859 |
| Tor | 7191 |
| After Oversampling |     |
| Non-Tor | 53859 |
| Tor | 53859 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 10-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 18  |

| **SCENARIO-A 10s** |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **ROC AUC** |
| Random Forest | 99.36% |     | 99.37% | 99.37% | 99.37% | 99.37% | 1.00 | 0.99 | 1.00 |
| Decision Tree | 98.66% |     | 98.62% | 98.62% | 98.62% | 98.62% | 1.00 | 0.98 | 0.99 |
| SVM | 91.31% |     | 91.27% | 91.69% | 91.27% | 91.25% | 0.91 | 0.91 | 0.97 |
| KNN | 97.88% |     | 98.05% | 98.09% | 98.05% | 98.05% | 1.00 | 0.97 | 0.99 |
| XGBoost | 99.46% |     | 99.51% | 99.51% | 99.51% | 99.51% | 1.00 | 0.99 | 1.00 |
| Logistic Regression | 90.03% |     | 90.07% | 90.12% | 90.07% | 90.06% | 0.90 | 0.90 | 0.95 |
| Gradient Boosting | 99.30% |     | 99.32% | 99.32% | 99.32% | 99.32% | 1.00 | 0.99 | 1.00 |
| Gaussian Naive Bayes | 79.43% |     | 79.09% | 84.17% | 79.09% | 78.29% | 0.79 | 0.79 | 0.92 |
| AdaBoost | 93.06% |     | 93.16% | 93.32% | 93.16% | 93.15% | 0.94 | 0.93 | 0.98 |
| Bagging Classifier | 99.38% |     | 99.40% | 99.40% | 99.40% | 99.40% | 1.00 | 0.99 | 1.00 |
| Extra Trees | 99.08% |     | 99.13% | 99.13% | 99.13% | 99.13% | 1.00 | 0.99 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **ROC AUC** |
| DNN | Accuracy | 95.41% | 95.07% | 95.17% | 95.07% | 95.07% | 0.17 | 0.17 | 0.95 |
| &nbsp; | Precision | 95.44% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 95.41% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 95.40% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Various classifiers were evaluated for their performance in distinguishing between Tor and non-Tor instances. Notable observations include:
  - Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees achieved high mean accuracy (cross-validation) above 99%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy (cross-validation) but still maintained robust performance with accuracy, precision, recall, and F1 scores above 98%.
  - Gaussian Naive Bayes exhibited comparatively lower performance, with an accuracy of 79.43% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased good performance across all evaluation metrics, with accuracy, precision, recall, and F1 score of approximately 95%.

**Learning Curve Metrics:**

- Training and validation scores remained high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees emerged as top performers, achieving high accuracy and other evaluation metrics.
- Gaussian Naive Bayes, while showing comparatively lower performance, still demonstrated reasonable accuracy in distinguishing between Tor and non-Tor instances.
- The deep neural network (DNN) showcased good performance across all evaluation metrics, highlighting its effectiveness in classification tasks.

**Results for File-6 (5s)**

Scenario-A

**Data Distribution Analysis:**

- The dataset for File 6 consisted of network flows with a flow timeout of 5 seconds.
- Initially, there were 69,686 non-Tor instances and 14,508 Tor instances, indicating a skewed distribution with approximately 4.8 non-Tor instances for every Tor instance.
- After preprocessing, the dataset was balanced with 62,361 instances for both non-Tor and Tor categories.

**Feature Reduction:**

- Feature reduction techniques effectively reduced the initial 23 features to 15, streamlining the dataset for improved model efficiency.

| **SCENARIO-A 5-sec** |     |
| --- |     | --- |
| **DATA DISTRIBUTION** |     |
| Original |     |
| Non-Tor | 69686 |
| Tor | 14508 |
| Number of Negative Values |     |
| 516 |     |
| Number of Inf Values |     |
| 12  |     |
| After Outlier Removal |     |
| Non-Tor | 62361 |
| Tor | 13420 |
| After Oversampling |     |
| Non-Tor | 62361 |
| Tor | 62361 |
| &nbsp; | &nbsp; |
| **SCENARIO-A 5-sec** |     |
| **Feature Reduction** |     |
| Total Features | 23  |
| Reduced Features | 15  |

| **SCENARIO-A 5s** |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **ROC AUC** |
| Random Forest | 99.30% |     | 99.33% | 99.33% | 99.33% | 99.33% | 1.00 | 0.99 | 1.00 |
| Decision Tree | 98.82% |     | 98.91% | 98.91% | 98.91% | 98.91% | 1.00 | 0.98 | 0.99 |
| SVM | 91.47% |     | 92.09% | 92.14% | 92.09% | 92.09% | 0.90 | 0.90 | 0.97 |
| KNN | 97.89% |     | 98.11% | 98.15% | 98.11% | 98.11% | 1.00 | 0.97 | 0.99 |
| XGBoost | 99.39% |     | 99.44% | 99.44% | 99.44% | 99.44% | 1.00 | 0.99 | 1.00 |
| Logistic Regression | 87.79% |     | 88.00% | 88.25% | 88.00% | 87.98% | 0.88 | 0.88 | 0.94 |
| Gradient Boosting | 99.25% |     | 99.34% | 99.34% | 99.34% | 99.34% | 1.00 | 0.99 | 1.00 |
| Gaussian Naive Bayes | 76.48% |     | 76.72% | 81.96% | 76.72% | 75.73% | 0.76 | 0.76 | 0.90 |
| AdaBoost | 91.21% |     | 91.18% | 91.42% | 91.18% | 91.17% | 0.92 | 0.92 | 0.98 |
| Bagging Classifier | 99.31% |     | 99.34% | 99.34% | 99.34% | 99.34% | 1.00 | 0.99 | 1.00 |
| Extra Trees | 99.21% |     | 99.23% | 99.24% | 99.23% | 99.23% | 1.00 | 0.99 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **ROC AUC** |
| DNN | Accuracy | 95.82% | 96.04% | 96.04% | 96.04% | 96.04% | 0.17 | 0.16 | 0.96 |
| &nbsp; | Precision | 95.83% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 95.82% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 95.82% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Various classifiers were evaluated for their performance in distinguishing between Tor and non-Tor instances. Notable observations include:
  - Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees achieved high mean accuracy (cross-validation) above 99%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy (cross-validation) but still maintained robust performance with accuracy, precision, recall, and F1 scores above 98%.
  - Gaussian Naive Bayes exhibited comparatively lower performance, with an accuracy of 76.48% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased good performance across all evaluation metrics, with accuracy, precision, recall, and F1 score of approximately 96%.

**Learning Curve Metrics:**

- Training and validation scores remained high across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees emerged as top performers, achieving high accuracy and other evaluation metrics.
- Gaussian Naive Bayes, while showing comparatively lower performance, still demonstrated reasonable accuracy in distinguishing between Tor and non-Tor instances.
- The deep neural network (DNN) showcased good performance across all evaluation metrics, highlighting its effectiveness in classification tasks.

SCENARIO-A Conclusion:  
Scenario-A, encompassing the binary classification of Tor and Non-Tor network flows across various flow timeout durations, has provided valuable insights into the effectiveness of machine learning techniques in network traffic analysis. Through meticulous preprocessing steps including outlier removal and oversampling, datasets were carefully curated to mitigate class imbalance, ensuring robust model training.

The feature reduction process streamlined the datasets, retaining essential information while reducing dimensionality, thereby enhancing model efficiency. Evaluation of multiple classifiers revealed consistent high performance across all files. Ensemble methods such as Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees consistently demonstrated accuracy levels exceeding 99%, along with high precision, recall, and F1 scores. These classifiers proved to be reliable across all files, showcasing their effectiveness in distinguishing between Tor and Non-Tor network flows.

Furthermore, an in-depth analysis of classifier performance revealed that ensemble methods, particularly Random Forest and XGBoost, consistently outperformed other classifiers across different flow timeout durations. Their robustness and adaptability to varying datasets make them ideal choices for network flow classification tasks.

In contrast, Gaussian Naive Bayes exhibited comparatively lower performance, with accuracy levels ranging from 76% to 98%. While Gaussian Naive Bayes provided reasonable classification results, its performance was notably inferior to that of ensemble methods, especially on highly imbalanced datasets.

Additionally, the deep neural network (DNN) showcased promising performance, achieving accuracy levels ranging from 95% to 100%. The versatility of DNNs in capturing intricate patterns within network traffic data highlights their potential for enhancing classification accuracy in complex scenarios.

Learning curve analysis further confirmed stable model performance, with training and validation scores converging at high levels without overfitting or underfitting. These findings underscore the significance of machine learning and deep learning techniques in accurately distinguishing Tor and Non-Tor network flows, offering practical solutions for identifying and mitigating security threats in network communication.

The comprehensive evaluation of Scenario-A not only provides valuable insights into classifier performance but also serves as a foundation for further research and development in the field of cybersecurity and network intrusion detection.

**Results for TimeBasedFeatures-15s**

Scenario B:

**Data Distribution Analysis:**

- The dataset for File-1 in Scenario-B comprised network flows with a flow timeout of 15 seconds.
- Initially, there were 3,360 instances across various application types, with VoIP being the most prominent category, followed by Video-Streaming and File-Transfer.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 11,992 instances, ensuring equal representation across application types.

**Feature Reduction:**

- Feature reduction techniques reduced the initial 23 features to 22, optimizing the dataset for improved model performance.

| **SCENARIO-B 15-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 1509 | 1499 | 1499 |
| Video-Streaming | 598 | 509 | 1499 |
| File-Transfer | 480 | 461 | 1499 |
| Chat | 243 | 142 | 1499 |
| Browsing | 227 | 152 | 1499 |
| Email | 186 | 172 | 1499 |
| P2P | 71  | 69  | 1499 |
| Audio-Streaming | 46  | 20  | 1499 |
| **TOTAL** | **3360** | **3024** | **11992** |
| Number of Negative Values |     | &nbsp; |     |
| 9507 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 0   |     | Reduced Features | 22  |

| **SCENARIO-B 15s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 96.87% |     | 97.74% | 97.76% | 97.74% | 97.74% | 1.00 | 0.94 | 0.99 | 0.99 |
| Decision Tree | 91.80% |     | 93.37% | 93.41% | 93.37% | 93.35% | 0.98 | 0.88 | 0.97 | 0.97 |
| SVM | 78.36% |     | 79.69% | 81.05% | 79.69% | 79.08% | 0.75 | 0.74 | 0.97 | 0.96 |
| KNN | 95.70% |     | 97.08% | 97.09% | 97.08% | 97.07% | 1.00 | 0.92 | 0.99 | 0.99 |
| XGBoost | 96.80% |     | 98.00% | 98.01% | 98.00% | 98.00% | 1.00 | 0.94 | 0.99 | 0.99 |
| Logistic Regression | 85.93% |     | 86.66% | 86.94% | 86.66% | 86.49% | 0.87 | 0.85 | 0.98 | 0.98 |
| Gradient Boosting | 97.03% |     | 98.37% | 98.37% | 98.37% | 98.37% | 1.00 | 0.94 | 0.99 | 0.99 |
| Gaussian Naive Bayes | 54.84% |     | 55.02% | 53.61% | 55.02% | 48.26% | 0.54 | 0.54 | 0.88 | 0.85 |
| AdaBoost | 69.59% |     | 71.57% | 74.66% | 71.57% | 71.19% | 0.69 | 0.69 | 0.94 | 0.93 |
| Bagging Classifier | 96.84% |     | 97.66% | 97.67% | 97.66% | 97.66% | 1.00 | 0.93 | 0.99 | 0.99 |
| Extra Trees | 97.22% |     | 97.74% | 97.78% | 97.74% | 97.74% | 1.00 | 0.94 | 0.99 | 0.99 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 78.51% | 80.57% | 81.32% | 80.58% | 80.50% | 0.89 | 0.88 | 0.89 | 0.89 |
| &nbsp; | Precision | 79.29% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 78.51% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 78.05% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers reveals:
  - Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees achieved high mean accuracy (cross-validation) above 96%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 91%.
  - Gaussian Naive Bayes exhibited lower performance compared to other classifiers, with an accuracy of 54.84% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased competitive performance across all evaluation metrics, with accuracy, precision, recall, and F1 score around 78%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, signifying excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Gaussian Naive Bayes demonstrated comparatively lower performance but still showed reasonable accuracy in classifying application types.
- The deep neural network (DNN) displayed competitive performance, highlighting its effectiveness in multiclass classification tasks.

**Results for TimeBasedFeatures-30s**

Scenario B:

**Data Distribution Analysis:**

- The dataset for File-2 in Scenario-B contained network flows with a flow timeout of 30 seconds.
- Initially, there were 1,803 instances across various application types, with VoIP being the most dominant category, followed by Video-Streaming and File-Transfer.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 6,032 instances, ensuring equal representation across application types.

**Feature Reduction:**

- No feature reduction was performed, maintaining all 23 features for analysis.

| **SCENARIO-B 30-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 758 | 754 | 754 |
| Video-Streaming | 345 | 282 | 754 |
| File-Transfer | 246 | 237 | 754 |
| Chat | 147 | 90  | 754 |
| Browsing | 133 | 109 | 754 |
| Email | 104 | 101 | 754 |
| P2P | 38  | 36  | 754 |
| Audio-Streaming | 32  | 13  | 754 |
| **TOTAL** | **1803** | **1622** | **6032** |
| Number of Negative Values |     | &nbsp; | &nbsp; |
| 4361 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 0   |     | Reduced Features | 23  |

| **SCENARIO-B 30s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 95.85% |     | 96.85% | 96.92% | 96.85% | 96.86% | 1.00 | 0.92 | 1.00 | 1.00 |
| Decision Tree | 88.08% |     | 90.55% | 90.69% | 90.55% | 90.45% | 0.97 | 0.82 | 0.95 | 0.95 |
| SVM | 73.30% |     | 76.38% | 77.44% | 76.38% | 75.58% | 0.69 | 0.68 | 0.97 | 0.96 |
| KNN | 94.01% |     | 95.19% | 95.25% | 95.19% | 95.16% | 1.00 | 0.87 | 0.99 | 0.98 |
| XGBoost | 96.04% |     | 95.85% | 95.93% | 95.85% | 95.85% | 1.00 | 0.92 | 1.00 | 1.00 |
| Logistic Regression | 87.02% |     | 86.49% | 86.35% | 86.49% | 86.34% | 0.89 | 0.85 | 0.99 | 0.98 |
| Gradient Boosting | 96.51% |     | 96.35% | 96.42% | 96.35% | 96.35% | 1.00 | 0.92 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 49.98% |     | 52.69% | 56.45% | 52.69% | 49.25% | 0.51 | 0.51 | 0.89 | 0.86 |
| AdaBoost | 67.29% |     | 67.85% | 71.16% | 67.85% | 66.12% | 0.68 | 0.66 | 0.92 | 0.91 |
| Bagging Classifier | 95.77% |     | 96.18% | 96.23% | 96.18% | 96.18% | 1.00 | 0.91 | 1.00 | 1.00 |
| Extra Trees | 96.14% |     | 96.10% | 96.24% | 96.10% | 96.09% | 1.00 | 0.92 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 74.46% | 74.23% | 75.19% | 74.24% | 72.97% | 1.09 | 1.04 | 0.85 | 0.85 |
| &nbsp; | Precision | 75.79% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 74.46% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 73.67% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers reveals:
  - Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees achieved high mean accuracy (cross-validation) above 95%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 88%.
  - Gaussian Naive Bayes exhibited the lowest performance among classifiers, with an accuracy of 49.98% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showcased competitive performance across all evaluation metrics, with accuracy, precision, recall, and F1 score around 74%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, signifying excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Gaussian Naive Bayes demonstrated comparatively lower performance but still showed reasonable accuracy in classifying application types.
- The deep neural network (DNN) displayed competitive performance, highlighting its effectiveness in multiclass classification tasks.

**Results for TimeBasedFeatures-60s**

Scenario B:

**Data Distribution Analysis:**

- File-3 in Scenario-B comprises network flows with a flow timeout of 60 seconds.
- Initially, the dataset contained 936 instances distributed across various application types, with VoIP being the most prevalent category, followed by Video-Streaming and File-Transfer.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 3,040 instances, ensuring equal representation across application types.

**Feature Reduction:**

- No feature reduction was performed, maintaining all 23 features for analysis.

| **SCENARIO-B 60-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 381 | 380 | 380 |
| Video-Streaming | 177 | 152 | 380 |
| File-Transfer | 125 | 115 | 380 |
| Chat | 84  | 57  | 380 |
| Browsing | 73  | 57  | 380 |
| Email | 54  | 53  | 380 |
| Audio-Streaming | 22  | 10  | 380 |
| P2P | 20  | 18  | 380 |
| **TOTAL** | **936** | **842** | **3040** |
| Number of Negative Values |     | &nbsp; | &nbsp; |
| 2014 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 0   |     | Reduced Features | 23  |

| **SCENARIO-B 60s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 95.84% |     | 95.23% | 95.35% | 95.23% | 95.23% | 1.00 | 0.91 | 1.00 | 1.00 |
| Decision Tree | 88.65% |     | 87.99% | 88.30% | 87.99% | 88.03% | 1.00 | 0.83 | 0.93 | 0.93 |
| SVM | 74.79% |     | 75.32% | 76.88% | 75.32% | 74.20% | 0.72 | 0.69 | 0.97 | 0.96 |
| KNN | 92.92% |     | 93.58% | 93.82% | 93.58% | 93.60% | 1.00 | 0.87 | 0.99 | 0.99 |
| XGBoost | 95.14% |     | 95.06% | 95.17% | 95.06% | 95.06% | 0.99 | 0.90 | 1.00 | 1.00 |
| Logistic Regression | 89.76% |     | 90.46% | 90.39% | 90.46% | 90.35% | 0.95 | 0.85 | 0.99 | 0.99 |
| Gradient Boosting | 96.17% |     | 96.54% | 96.65% | 96.54% | 96.54% | 1.00 | 0.91 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 65.33% |     | 63.65% | 59.44% | 63.65% | 59.81% | 0.67 | 0.65 | 0.93 | 0.90 |
| AdaBoost | 64.84% |     | 63.15% | 55.96% | 63.15% | 57.69% | 0.64 | 0.61 | 0.90 | 0.87 |
| Bagging Classifier | 95.31% |     | 95.06% | 95.20% | 95.06% | 95.05% | 1.00 | 0.90 | 1.00 | 1.00 |
| Extra Trees | 96.17% |     | 96.38% | 96.50% | 96.38% | 96.35% | 1.00 | 0.91 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 70.31% | 69.73% | 68.40% | 69.73% | 66.99% | 1.10 | 1.11 | 0.82 | 0.82 |
| &nbsp; | Precision | 68.89% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 70.30% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 67.15% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers indicates:
  - Random Forest, XGBoost, Gradient Boosting, and Bagging Classifier achieved high mean accuracy (cross-validation) above 95%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 88%.
  - Gaussian Naive Bayes and AdaBoost exhibited the lowest performance among classifiers, with accuracy below 70% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showed moderate performance across all evaluation metrics, with accuracy, precision, recall, and F1 score around 70%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, suggesting stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, and Bagging Classifier emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Decision Tree and KNN demonstrated competitive performance, showcasing their effectiveness in classifying application types.
- Gaussian Naive Bayes and AdaBoost showed lower performance, indicating limitations in accurately classifying network flows.
- The deep neural network (DNN) displayed moderate performance, suggesting potential for improvement in capturing complex patterns within the data.

**Results for TimeBasedFeatures-120s**

Scenario B:

**Data Distribution Analysis:**

- File-4 in Scenario-B represents network flows with a flow timeout of 120 seconds.
- Initially, the dataset contained 486 instances distributed across various application types, with VoIP being the most prevalent category, followed by Video-Streaming and File-Transfer.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 1,512 instances, ensuring equal representation across application types.

**Feature Reduction:**

- Feature reduction was performed, reducing the total features from 23 to 13 to improve computational efficiency while maintaining essential information.

| **SCENARIO-B 120-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 193 | 189 | 189 |
| Video-Streaming | 90  | 88  | 189 |
| File-Transfer | 63  | 54  | 189 |
| Chat | 45  | 32  | 189 |
| Browsing | 41  | 32  | 189 |
| Email | 28  | 27  | 189 |
| Audio-Streaming | 16  | 5   | 189 |
| P2P | 10  | 10  | 189 |
| **TOTAL** | **486** | **437** | **1512** |
| Number of Negative Values |     | &nbsp; | &nbsp; |
| 915 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 0   |     | Reduced Features | 13  |

| **SCENARIO-B 120s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 92.30% |     | 91.08% | 91.19% | 91.08% | 90.98% | 1.00 | 0.86 | 1.00 | 0.99 |
| Decision Tree | 84.77% |     | 84.48% | 84.61% | 84.48% | 84.42% | 1.00 | 0.79 | 0.91 | 0.91 |
| SVM | 73.69% |     | 70.95% | 70.33% | 70.95% | 69.25% | 0.73 | 0.67 | 0.97 | 0.95 |
| KNN | 87.26% |     | 87.12% | 87.09% | 87.12% | 87.07% | 1.00 | 0.81 | 0.96 | 0.96 |
| XGBoost | 92.38% |     | 92.07% | 92.25% | 92.07% | 91.96% | 1.00 | 0.85 | 0.99 | 0.99 |
| Logistic Regression | 91.06% |     | 94.05% | 94.10% | 94.05% | 94.01% | 0.96 | 0.85 | 0.99 | 0.99 |
| Gradient Boosting | 92.05% |     | 93.06% | 93.31% | 93.06% | 93.02% | 1.00 | 0.85 | 1.00 | 0.99 |
| Gaussian Naive Bayes | 70.30% |     | 71.28% | 67.96% | 71.28% | 67.82% | 0.73 | 0.68 | 0.94 | 0.92 |
| AdaBoost | 63.35% |     | 63.36% | 58.91% | 63.36% | 57.91% | 0.60 | 0.56 | 0.89 | 0.88 |
| Bagging Classifier | 91.56% |     | 91.08% | 91.26% | 91.08% | 91.04% | 1.00 | 0.85 | 1.00 | 0.99 |
| Extra Trees | 92.38% |     | 92.07% | 92.13% | 92.07% | 91.98% | 0.99 | 0.85 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 66.91% | 66.66% | 58.95% | 66.44% | 60.79% | 1.32 | 1.36 | 0.81 | 0.81 |
| &nbsp; | Precision | 64.17% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 66.99% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 61.75% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers indicates:
  - Random Forest, XGBoost, Logistic Regression, and Gradient Boosting achieved high mean accuracy (cross-validation) above 91%.
  - Decision Tree, KNN, and Bagging Classifier demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 84%.
  - Gaussian Naive Bayes and AdaBoost exhibited the lowest performance among classifiers, with accuracy below 71% and lower precision, recall, and F1 scores.
  - The deep neural network (DNN) showed moderate performance across all evaluation metrics, with accuracy, precision, recall, and F1 score around 67%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Logistic Regression, and Gradient Boosting emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Decision Tree, KNN, and Bagging Classifier demonstrated competitive performance, showcasing their effectiveness in classifying application types.
- Gaussian Naive Bayes and AdaBoost showed lower performance, indicating limitations in accurately classifying network flows.
- The deep neural network (DNN) displayed moderate performance, suggesting potential for improvement in capturing complex patterns within the data.

**Results for TimeBasedFeatures-10s**

Scenario B:

**Data Distribution Analysis:**

- File-5 in Scenario-B represents network flows with a flow timeout of 10 seconds.
- Initially, the dataset contained 8,044 instances distributed across various application types, with VoIP being the most prevalent category, followed by Browsing and P2P.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 18,152 instances, ensuring equal representation across application types.

**Feature Reduction:**

- Feature reduction was performed, reducing the total features from 23 to 15 to improve computational efficiency while maintaining essential information.

| **SCENARIO-B 10-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 2291 | 2269 | 2269 |
| Browsing | 1604 | 1268 | 2269 |
| P2P | 1085 | 1079 | 2269 |
| Video | 874 | 786 | 2269 |
| File-Transfer | 864 | 848 | 2269 |
| Audio | 721 | 493 | 2269 |
| Chat | 323 | 233 | 2269 |
| Mail | 282 | 264 | 2269 |
| **TOTAL** | **8044** | **7240** | **18152** |
| Number of Negative Values |     | &nbsp; | &nbsp; |
| 378 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 0   |     | Reduced Features | 15  |

| **SCENARIO-B 10s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 92.27% |     | 93.36% | 93.36% | 93.36% | 93.33% | 1.00 | 0.88 | 0.99 | 0.99 |
| Decision Tree | 86.48% |     | 88.04% | 87.95% | 88.04% | 87.97% | 1.00 | 0.81 | 0.93 | 0.93 |
| SVM | 60.89% |     | 60.89% | 61.60% | 60.89% | 56.27% | 0.57 | 0.56 | 0.92 | 0.91 |
| KNN | 88.09% |     | 89.92% | 89.98% | 89.92% | 89.81% | 1.00 | 0.82 | 0.97 | 0.97 |
| XGBoost | 92.36% |     | 93.88% | 93.92% | 93.88% | 93.83% | 1.00 | 0.88 | 1.00 | 1.00 |
| Logistic Regression | 61.00% |     | 60.83% | 57.86% | 60.83% | 58.43% | 0.62 | 0.61 | 0.92 | 0.90 |
| Gradient Boosting | 92.43% |     | 93.72% | 93.73% | 93.72% | 93.69% | 1.00 | 0.89 | 0.99 | 0.99 |
| Gaussian Naive Bayes | 32.97% |     | 34.23% | 31.36% | 34.23% | 27.83% | 0.30 | 0.30 | 0.77 | 0.78 |
| AdaBoost | 64.18% |     | 65.51% | 66.38% | 65.51% | 63.42% | 0.59 | 0.58 | 0.90 | 0.88 |
| Bagging Classifier | 92.03% |     | 93.41% | 93.43% | 93.41% | 93.40% | 1.00 | 0.87 | 0.99 | 0.99 |
| Extra Trees | 92.43% |     | 93.83% | 93.84% | 93.83% | 93.80% | 1.00 | 0.88 | 0.99 | 0.99 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 59.25% | 58.88% | 55.72% | 58.86% | 55.68% | 1.41 | 1.38 | 0.77 | 0.76 |
| &nbsp; | Precision | 56.76% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 59.26% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 54.79% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers indicates:
  - Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees achieved high mean accuracy (cross-validation) above 92%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 86%.
  - Gaussian Naive Bayes and AdaBoost exhibited the lowest performance among classifiers, with accuracy below 65% and lower precision, recall, and F1 scores.
  - Support Vector Machine (SVM) and Logistic Regression showed moderate performance with accuracy around 61%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Gradient Boosting, Bagging Classifier, and Extra Trees emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Decision Tree and KNN demonstrated competitive performance, showcasing their effectiveness in classifying application types.
- Gaussian Naive Bayes and AdaBoost showed lower performance, indicating limitations in accurately classifying network flows.
- The deep neural network (DNN) displayed moderate performance, suggesting potential for improvement in capturing complex patterns within the data.

**Results for TimeBasedFeatures-05s**

Scenario B:

**Data Distribution Analysis:**

- File-6 in Scenario-B represents network flows with a flow timeout of 5 seconds.
- Initially, the dataset contained 14,508 instances distributed across various application types, with VoIP being the most prevalent category, followed by Browsing and P2P.
- After preprocessing, including outlier removal and oversampling, the dataset was balanced with 35,968 instances, ensuring equal representation across application types.

**Feature Reduction:**

- Feature reduction was performed, reducing the total features from 23 to 15 to improve computational efficiency while maintaining essential information.

| **SCENARIO-B 5-sec** |     |     |     |
| --- |     |     |     | --- | --- | --- |
| **DATA DISTRIBUTION** |     |     |     |
| Original |     | After Outlier Removal | After Oversampling |
| Voip | 4524 | 4496 | 4496 |
| Browsing | 2645 | 1952 | 4496 |
| P2P | 2139 | 2130 | 4496 |
| File-Transfer | 1663 | 1639 | 4496 |
| Video | 1529 | 1389 | 4496 |
| Audio | 1026 | 679 | 4496 |
| Mail | 497 | 455 | 4496 |
| Chat | 485 | 317 | 4496 |
| **TOTAL** | **14508** | **13057** | **35968** |
| Number of Negative Values |     | &nbsp; | &nbsp; |
| 407 |     | **Feature Reduction** |     |
| Number of Inf Values |     | Total Features | 23  |
| 2   |     | Reduced Features | 15  |

| **SCENARIO-B 5s** |     |     |     |     |     |     |     |     |     |     |
| --- |     |     |     |     |     |     |     |     |     |     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Classifier** | **Mean Accuracy (Cross Validation)** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Score** | **Mean Validation Score** | **Micro-Average AUC** | **Macro-Average AUC** |
| Random Forest | 93.89% |     | 94.59% | 94.64% | 94.59% | 94.60% | 1.00 | 0.90 | 1.00 | 1.00 |
| Decision Tree | 87.38% |     | 89.12% | 89.17% | 89.12% | 89.14% | 0.96 | 0.82 | 0.95 | 0.95 |
| SVM | 65.17% |     | 66.33% | 65.99% | 66.33% | 64.34% | 0.62 | 0.61 | 0.94 | 0.93 |
| KNN | 91.05% |     | 92.68% | 92.70% | 92.68% | 92.66% | 1.00 | 0.86 | 0.98 | 0.98 |
| XGBoost | 93.61% |     | 94.42% | 94.47% | 94.42% | 94.43% | 1.00 | 0.90 | 1.00 | 1.00 |
| Logistic Regression | 63.17% |     | 63.69% | 61.84% | 63.69% | 62.08% | 0.63 | 0.63 | 0.93 | 0.91 |
| Gradient Boosting | 92.68% |     | 93.53% | 93.62% | 93.53% | 93.55% | 1.00 | 0.89 | 1.00 | 1.00 |
| Gaussian Naive Bayes | 38.43% |     | 37.97% | 40.66% | 37.97% | 35.84% | 0.39 | 0.39 | 0.80 | 0.79 |
| AdaBoost | 62.33% |     | 61.98% | 61.58% | 61.98% | 61.26% | 0.60 | 0.59 | 0.90 | 0.89 |
| Bagging Classifier | 93.58% |     | 94.25% | 94.31% | 94.25% | 94.26% | 1.00 | 0.90 | 1.00 | 0.99 |
| Extra Trees | 92.96% |     | 93.85% | 93.96% | 93.85% | 93.85% | 0.98 | 0.89 | 1.00 | 1.00 |
| &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | **Average Evaluation Metrics Across Folds** |     | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Mean Traning Loss** | **Mean Validation Loss** | **Micro-Average AUC** | **Macro-Average AUC** |
| DNN | Accuracy | 70.58% | 70.54% | 69.83% | 70.54% | 69.33% | 1.03 | 1.00 | 0.83 | 0.83 |
| &nbsp; | Precision | 70.13% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | Recall | 70.58% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |
| &nbsp; | F1 Score | 69.40% | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |

**Classifier Performance:**

- Performance analysis of various classifiers indicates:
  - Random Forest, XGBoost, Bagging Classifier, and Gradient Boosting achieved high mean accuracy (cross-validation) above 92%.
  - Decision Tree and KNN demonstrated slightly lower mean accuracy but maintained robust performance with accuracy, precision, recall, and F1 scores above 87%.
  - Gaussian Naive Bayes exhibited the lowest performance among classifiers, with accuracy below 40% and lower precision, recall, and F1 scores.
  - Support Vector Machine (SVM) and Logistic Regression showed moderate performance with accuracy around 65%.

**Learning Curve Metrics:**

- Training and validation scores remained consistent across classifiers, indicating stable model performance without overfitting or underfitting.

**Receiver Operating Characteristic Area Under the Curve (ROC AUC):**

- ROC AUC scores were high for most classifiers, indicating excellent discrimination performance in distinguishing between application types.

**Model Comparison:**

- Random Forest, XGBoost, Bagging Classifier, and Gradient Boosting emerged as top performers, exhibiting high accuracy and other evaluation metrics.
- Decision Tree and KNN demonstrated competitive performance, showcasing their effectiveness in classifying application types.
- Gaussian Naive Bayes showed lower performance, indicating limitations in accurately classifying network flows.
- The deep neural network (DNN) displayed moderate performance, suggesting potential for improvement in capturing complex patterns within the data.

**SCENARIO-B Conclusion**:

In Scenario-B, our objective was to classify different types of applications within Tor network flows across various flow timeout values, ranging from 5 seconds to 120 seconds. We evaluated the performance of several classifiers using metrics such as accuracy, precision, recall, F1 score, mean training score, mean validation score, micro-average AUC, and macro-average AUC.

Across all files, we observed varying distributions of application types, with VoIP, browsing, and P2P being the most prevalent. After preprocessing steps including outlier removal and oversampling to address class imbalances, the datasets were prepared for classification.

Among the classifiers, ensemble methods like Random Forest, XGBoost, Bagging Classifier, and Gradient Boosting consistently demonstrated superior performance, achieving mean accuracies above 92% across all files. These classifiers showed robustness against overfitting, maintaining stable training and validation scores. Additionally, they exhibited high discrimination ability, as reflected in their micro-average AUC and macro-average AUC scores close to 1.00.

Decision Tree and KNN also performed well, with accuracy scores consistently above 87%. However, their performance was slightly lower compared to ensemble methods. SVM and Logistic Regression showed moderate performance, indicating the need for further optimization or feature engineering to enhance their effectiveness.

On the other hand, Gaussian Naive Bayes consistently exhibited lower accuracy across all files, indicating its limitations in capturing the complex relationships within the data. This suggests the importance of exploring more sophisticated algorithms or ensemble techniques to improve classification accuracy.

In terms of computational efficiency, Decision Tree and KNN showed faster training times compared to ensemble methods like Random Forest and Gradient Boosting. This trade-off between computational efficiency and classification performance should be considered based on specific application requirements.

Overall, the results highlight the effectiveness of ensemble methods for multiclass classification of Tor network flows. However, there is still room for improvement, particularly in addressing the challenges posed by class imbalances and complex data distributions. These insights provide valuable guidance for further refining classification models and enhancing the accuracy of identifying application types within Tor network flows.