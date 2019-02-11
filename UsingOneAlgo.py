import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class train:

    dataframe = ""
    featuretrain = ""
    labeltrain = ""
    featuretest = ""
    labeltest = ""
    predictionlabel = ""
    finalresult = pd.DataFrame()
    richpeople = pd.DataFrame()
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

            clf2 = RandomForestClassifier(criterion = "gini", random_state = 60, max_depth = 20, n_estimators = 200)
            clf2.fit(self.featuretrain,self.labeltrain)
            self.predictiontest = clf2.predict(self.featuretest)
            predictiontest2 = clf2.predict_proba(self.featuretest)
            
            result = []
            j = 0
            for i in predictiontest2:
                if i[1] > 0.63000000:
                    result.append(j)
                j+=1
            pd.DataFrame(result).to_csv("Rich_people.csv")
            self.no+=1
            self.displayAccuracy("RandomForestClassifier")
                    
    def displayAccuracy(self, name):

        counf = confusion_matrix(self.labeltest, self.predictiontest)
        self.score = self.score+(counf[0][0]+counf[1][1])
        print ("Score is:")
        print (((self.score)/self.length)*100)
        exit()
            
        
T = train()
T.loadData('train.csv', 'test.csv')
T.fillNaN()
T.cleaningData()
T.splitData()
T.cleanDataFrame()
print ("Feeding data\n")
T.trainingModels()


