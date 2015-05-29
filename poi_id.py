#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import tree, preprocessing, linear_model

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'expenses', 'exercised_stock_options', 'other', 'from_this_person_to_poi'] 
# You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
## Fix the two bad data lines
## Has to do with misreading based on order of features in the PDF file (female was added on as it doesn't matter)
ordered_feature_list = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 
                        'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 
                        'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
                        'female']
for i in range(3,13):
    data_dict["BELFER ROBERT"][ordered_feature_list[i]] = data_dict["BELFER ROBERT"][ordered_feature_list[i+1]]
data_dict["BELFER ROBERT"][ordered_feature_list[13]] = 'NaN'
for i in range(len(ordered_feature_list)-1, 4, -1):
    data_dict["BHATNAGAR SANJAY"][ordered_feature_list[i]] =  data_dict["BHATNAGAR SANJAY"][ordered_feature_list[i-1]]

# AT THE END OF THIS SHOULD HAVE FIXED data_dict

### Task 3: Create new feature(s)
# add new feature "female"
female = ['BECK SALLY W', 'CARTER REBECCA C', 'DIETRICH JANET R', 
          'FOWLER PEGGY', 'GRAMM WENDY L', 'JACKSON CHARLENE R', 
          'KITCHEN LOUISE', 'MARTIN AMANDA K', 'MCDONALD REBECCA',
          'MORDAUN, KRISTINA M', 'MURRAY JULIA H', 'OLSON CINDY K', 
          'RIEKER PAULA H', 'SHAR, VICTORIA T',
          'SULLIVAN-SHAKLOVITZ COLLEEN', 'TILNEY ELIZABETH A']
for k, v in data_dict.iteritems():
    if k in female:
        v['female'] = 1
    else:
        v['female'] = 0

# add new feature "has_email_address"
#(having a non-numeric feature email_address doesn't seem to help us much here)
for k, v in data_dict.iteritems():
    if v['email_address'] != 'NaN':
        v['has_email_address'] = 1
    else:
        v['has_email_address'] = 0
    

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Uncomment this code block to test different models
# It was slowing down my run, so I commented it out
for clf, name in ((GaussianNB(), "GaussianNB"),
      (RandomForestClassifier(n_estimators=100), "Random Forest"),
      (linear_model.Perceptron(), "Perceptron"), 
      (tree.DecisionTreeClassifier(), "Decision Tree" )): 
      test_classifier(clf, my_dataset, features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# TUNING WAS DONE IN IPYTHON - DO YOU NEED TO SEE CODE?
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features='auto', max_leaf_nodes=None, min_samples_leaf=3,
            min_samples_split=8, min_weight_fraction_leaf=0.0,
            random_state=100, splitter='best')
test_classifier(clf, my_dataset, features_list)
print features_list, clf.feature_importances_
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)