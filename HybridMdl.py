import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class train:

    dataframe = ""
    featuretrain = ""
    labeltrain = ""
    featuretest = ""
    labeltest = ""
    predictionlabel = ""
    finalresult = pd.DataFrame()
    score = 0
    length = 0
    no = -1
    
    def loadData(self, pathtrain, pathtest):

        self.dataframe = pd.read_csv(pathtrain, skipinitialspace=True, encoding = "utf-8", names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hour-per-week","native-country","salary"])
        self.dataframe = pd.DataFrame(self.dataframe, columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hour-per-week","native-country","salary"])        
        self.dataframetest = pd.read_csv(pathtest,  skipinitialspace=True, encoding = "utf-8", names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hour-per-week","native-country","salary"])
        self.dataframetest = pd.DataFrame(self.dataframetest, columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hour-per-week","native-country","salary"])
        self.length = len(self.dataframetest)
        print("Data Loaded Successfully\n")

    def fillNaN(self):

        self.dataframe = self.dataframe.replace('?', float('nan'))
        self.dataframe = self.dataframe.replace(' ', '')
        self.dataframetest = self.dataframetest.replace('?', float('nan'))
        self.dataframetest = self.dataframetest.replace(' ', '')
        print("Replace null values\n")

    def cleaningData(self):

        self.dataframe['workclass'] = self.dataframe['workclass'].fillna('Private')
        self.dataframe['native-country'] = self.dataframe['native-country'].fillna('United-States')
        self.dataframe['occupation'] = self.dataframe['occupation'].fillna(method = "bfill")

        self.dataframetest['workclass'] = self.dataframetest['workclass'].fillna('Private')
        self.dataframetest['native-country'] = self.dataframetest['native-country'].fillna('United-States')
        self.dataframetest['occupation'] = self.dataframetest['occupation'].fillna(method = "bfill")
        print ("Cleaning data\n")

    def cleanDataFrame(self):

        for i in self.featuretrain:
            if type(self.featuretrain[i][3]) == str:
                
                out = set(self.featuretrain[i])
                cnt = 1
                for i2 in out:
                    self.featuretrain[i] = self.featuretrain[i].replace(i2, cnt)
                    self.featuretest[i] = self.featuretest[i].replace(i2, cnt)
                    cnt+=1
                
            else:
                pass

        self.labeltrain = self.labeltrain.replace("<=50K", 0)
        self.labeltest = self.labeltest.replace(">50K.", 1)
        self.labeltest = self.labeltest.replace("<=50K.", 0)
        self.labeltrain = self.labeltrain.replace(">50K", 1)
        print ("Prepared for feed\n")

    def splitData(self):

        self.labeltrain = self.dataframe['salary']
        self.featuretrain = self.dataframe.drop(columns=['salary','fnlwgt'])
        self.dataframe = self.dataframe.drop(columns = ['fnlwgt'])
        self.labeltest = self.dataframetest['salary']
        self.featuretest = self.dataframetest.drop(columns=['salary','fnlwgt'])
        self.dataframetest = self.dataframetest.drop(columns = ['fnlwgt'])
        
    def trainingModels(self):

        if self.no == -1:
            clf2 = RandomForestClassifier(criterion = "gini", random_state = 60, max_depth = 20, n_estimators = 200)
            clf2.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf2.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("RandomForestClassifier")
            
        elif self.no == 0:
            clf5 = GradientBoostingClassifier()
            clf5.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf5.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("GradientBoostingClassifier")

        elif self.no == 1:
            clf4 = AdaBoostClassifier()
            clf4.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf4.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("AdaBoostClassifier")

        elif self.no == 2:
            clf9 = LinearDiscriminantAnalysis()
            clf9.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf9.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("LinearDiscriminantAnalysis")

        elif self.no == 3:
            clf10 = QuadraticDiscriminantAnalysis()
            clf10.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf10.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("QuadraticDiscriminantAnalysis")

        elif self.no == 4:
            clf3 = DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth = 20, min_samples_leaf=10)
            clf3.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf3.predict(self.featuretest)
            self.no+=1
            self.displayAccuracy("DecisionTreeClassifier")

        elif self.no == 5:
            print ("Score is:")
            print (((self.score)/16281)*100)
            self.finalresult.to_csv("Predicted_result.csv", sep = ',', encoding = 'utf-8')
            exit()
                    
    def displayAccuracy(self, name):

        bad = []
        corr = []
        j1 = self.labeltest.index
        j = 0
        for i in self.labeltest:
            if self.predictiontest[j] != i:
                bad.append(j1[j])

            else:
                corr.append(j1[j])

            j+=1
        counf = confusion_matrix(self.labeltest, self.predictiontest)
        self.score = self.score+(counf[0][0]+counf[1][1])
        prob = (self.score)/self.length
        self.finalresult = self.finalresult.append(self.dataframetest.drop(bad))
        self.dataframetest = self.dataframetest.drop(corr)
        self.featuretest = self.featuretest.loc[bad]
        self.labeltest = self.labeltest.loc[bad]
        self.trainingModels()
        
T = train()
T.loadData('train.csv', 'test.csv')
T.fillNaN()
T.cleaningData()
T.splitData()
T.cleanDataFrame()
print ("Feeding data\n")
T.trainingModels()


