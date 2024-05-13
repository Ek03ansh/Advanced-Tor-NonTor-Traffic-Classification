# Advanced-Tor-NonTor-Traffic-Classification
This repository contains the detailed results for our research paper *Enhancing Tor-NonTor Traffic Classification Across Diverse Classifiers*. The **Results** Folder contains the results.

The Results Folder consists of two main subfolders: **Scenario-A** and **Scenario-B**. Each subfolder contains files corresponding to different flow-timeout settings.

| Flow Timeout | Scenario-A File                            | Scenario-B File                            |
|--------------|--------------------------------------------|--------------------------------------------|
| 5 seconds   | [Scenario-A-05s](Results/Scenario-A/Scenario-A-05s.md)| [Scenario-B-05s](Results/Scenario-B/Scenario-B-05s)|
| 10 seconds   | [Scenario-A-10s](Results/Scenario-A/Scenario-A-10s.md)| [Scenario-B-10s](Results/Scenario-B/Scenario-B-10s)|
| 15 seconds   | [Scenario-A-15s](Results/Scenario-A/Scenario-A-15s.md)| [Scenario-B-15s](Results/Scenario-B/Scenario-B-15s)|
| 30 seconds   | [Scenario-A-30s](Results/Scenario-A/Scenario-A-30s.md)| [Scenario-B-30s](Results/Scenario-B/Scenario-B-30s)|
| 60 seconds   | [Scenario-A-60s](Results/Scenario-A/Scenario-A-60s.md)| [Scenario-B-60s](Results/Scenario-B/Scenario-B-60s)|
| 120 seconds  | [Scenario-A-120s](Results/Scenario-A/Scenario-A-120s.md)| [Scenario-B-120s](Results/Scenario-B/Scenario-B-120s)|


## SCENARIO-A Results Conclusion
**Scenario-A**, focusing on classifying Tor and Non-Tor network flows across various timeout durations, employed meticulous preprocessing steps to curate datasets, addressing class imbalance via *outlier removal* and *oversampling*. Feature reduction enhanced model efficiency by retaining essential information while reducing dimensionality.

Ensemble methods like *Random Forest*, *XGBoost*, and *Gradient Boosting* consistently achieved over **99% accuracy**, along with high precision, recall, and F1 scores, proving reliable in distinguishing network flows. Notably, **Random Forest** and **XGBoost** outperformed other classifiers across different timeout durations.

*Gaussian Naive Bayes* showed lower performance (*76%-98%*) compared to ensemble methods, especially on imbalanced datasets. **Deep Neural Networks (DNNs)** exhibited promising accuracy (*95%-100%*), showcasing potential in capturing intricate patterns.

Learning curve analysis indicated stable model performance without overfitting or underfitting, emphasizing the significance of *machine learning* and *deep learning* techniques in network flow classification for cybersecurity. This evaluation lays a foundation for further research in network intrusion detection.


## SCENARIO-B Results Conclusion

In **Scenario-B**, our objective was to classify different types of applications within Tor network flows across various flow timeout values, ranging from 5 seconds to 120 seconds. We evaluated the performance of several classifiers using metrics such as *accuracy*, *precision*, *recall*, *F1 score*, *mean training score*, *mean validation score*, *micro-average AUC*, and *macro-average AUC*.

Across all files, we observed varying distributions of application types, with **VoIP**, **Browsing**, and **P2P** being the most prevalent. After preprocessing steps including *outlier removal* and *oversampling* to address class imbalances, the datasets were prepared for classification.

Among the classifiers, ensemble methods like **Random Forest**, **XGBoost**, **Bagging Classifier**, and **Gradient Boosting** consistently demonstrated superior performance, achieving mean accuracies above **92%** across all files. These classifiers showed robustness against overfitting, maintaining stable training and validation scores. Additionally, they exhibited high discrimination ability, as reflected in their micro-average AUC and macro-average AUC scores close to **1.00**.

**Decision Tree** and **KNN** also performed well, with accuracy scores consistently above **87%**. However, their performance was slightly lower compared to ensemble methods. **SVM** and **Logistic Regression** showed moderate performance, indicating the need for further optimization or feature engineering to enhance their effectiveness.

On the other hand, **Gaussian Naive Bayes** consistently exhibited lower accuracy across all files, indicating its limitations in capturing the complex relationships within the data. This suggests the importance of exploring more sophisticated algorithms or ensemble techniques to improve classification accuracy.

In terms of computational efficiency, **Decision Tree** and **KNN** showed faster training times compared to ensemble methods like **Random Forest** and **Gradient Boosting**. This trade-off between computational efficiency and classification performance should be considered based on specific application requirements.

Overall, the results highlight the effectiveness of ensemble methods for multiclass classification of Tor network flows. However, there is still room for improvement, particularly in addressing the challenges posed by class imbalances and complex data distributions. These insights provide valuable guidance for further refining classification models and enhancing the accuracy of identifying application types within Tor network flows.

