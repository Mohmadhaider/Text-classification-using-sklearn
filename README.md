# Text-classification-using-sklearn
 Predict whether income exceeds $50K/yr based on census data

### Prerequisites
 It require python installed on your machine. You also need to install following dependence.
 
 ```
 pip install scikit-learn
 pip install pandas
 pip install numpy
 ```
 
 ## Files
 
 ### HybridMdl
  This file will use sklearn's following algorithms to predict the test samples correctly.
   * RandomForestClassifier
   * GradientBoostingClassifier
   * AdaBoostClassifier
   * LinearDiscriminantAnalysis
   * QuadraticDiscriminantAnalysis
   * DecisionTreeClassifier
   
  The accuracy noted is 93%.
  
  ### UsingOneAlgo
   This file will use RandomForestClassifier to pridict the salary of peoples.
    
   The accuracy noted is 86%.
   
  ## Running the tests
   * Clone repo to your local drive.
   * Extract zip file.
   * Execute HybridMdl.py or UsingOneAlgo by following command.
    ```
    python HybridMdl.py
    python UsingOneAlgo.py
    ```
 
