# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-08T13:31:16.803845Z","iopub.execute_input":"2024-04-08T13:31:16.804233Z","iopub.status.idle":"2024-04-08T13:31:19.292682Z","shell.execute_reply.started":"2024-04-08T13:31:16.804204Z","shell.execute_reply":"2024-04-08T13:31:19.291199Z"}}
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from sklearn . preprocessing import OrdinalEncoder
import sklearn.model_selection as model_selection
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import sensitivity_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn . metrics import accuracy_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from imblearn.metrics import specificity_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn . preprocessing import MinMaxScaler
from sklearn . model_selection import train_test_split
from sklearn . metrics import classification_report
from sklearn import tree
from sklearn . neighbors import KNeighborsClassifier
from sklearn . metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn . model_selection import GridSearchCV
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn . linear_model import LogisticRegression
from matplotlib.ticker import FormatStrFormatter
from sklearn.cluster import KMeans
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
missing_values = ["n.a.","?","NA","n/a", "na"," --"]
columns = ["1. What is your biggest reason for learning to code?", "7. About how many hours do you spend learning each week?", "8. About how many months have you been programming?", "22. About how much money did you earn last year from any job or employment (in US Dollars)? ","23. How old are you?", "24. Which of the following best represents how you think of yourself?", "32. What is the highest degree or level of school you have completed?"]
q1columns = ["To change careers", "To start your first career", "To start a business or to freelance", "To succeed in current career","As a hobby"]
gend = ['Male','Female','Nonbinary','Prefer not to say']
month = ['6', '24', '50', '36', '30', '12', '120', '48', '7', '14', '3',
       '18', '9', '4', '52', '8', '10', '2', '26', '13', '72', '240',
       '15', '1', '60', '150', '11', '23', '180', '16', '20', '5', '28',
       '55', '38', '300', '99', '27', '40', '80', '47', '105', '17', '0',
       '70', '19', '39', '69', '35', '33', '32', '63', '42', '100', '96',
       '25', '21', '54', '31', '43', '216', '88', '34', '190','400', '75', '360', '57', '204', '45', '58', '350', '276', '22',
       '56','01','29', '200', '0.5', '1.5', '4.5', '2.5', '288', '94', '0.2', '000',
       '420', '128', '144', '90', '68', '5.5', '0.75', '84', '6.5', '02',
       '44', '250', '0.25', '46', '168', '0.03', '0.7',
       '156', '252', '78', '220', '06', '132', '04', '108', '432', '492',
        '480', '49', '00', '110', '257', '140', '134',
       '77', '0.8', '92', '109', '118', '09', '336', '85',
       '71', '104', '66', '160', '4.7', '130', '500', '37', '148',
       '192', '0.4', '1.2', '396', '65', '264', '121', '129']
education = ['Bachelor’s degree', 'Some college credit, no degree',
       'Professional degree (MBA, MD, JD, etc.)',
       "Master's degree (non-professional)", 'Associate’s degree',
       'No high school (secondary school)',
       'High school diploma or equivalent (GED)',
       'Trade, technical, or vocational training', 'Ph.D.',
       'Some high school']
clusters = ['0', '2', '1', '3']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-08T13:31:19.294745Z","iopub.execute_input":"2024-04-08T13:31:19.295221Z","iopub.status.idle":"2024-04-08T13:31:20.177671Z","shell.execute_reply.started":"2024-04-08T13:31:19.295192Z","shell.execute_reply":"2024-04-08T13:31:20.176248Z"}}
data = pd.read_csv("/kaggle/input/2021-new-coder-survey/2021 New Coder Survey.csv", na_values = missing_values, low_memory=False)
pd.set_option('display.max_rows', 100)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-04-08T13:31:20.179478Z","iopub.execute_input":"2024-04-08T13:31:20.179886Z","iopub.status.idle":"2024-04-08T13:31:20.215568Z","shell.execute_reply.started":"2024-04-08T13:31:20.179852Z","shell.execute_reply":"2024-04-08T13:31:20.214608Z"}}
df2 = data[columns]
df2.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.217892Z","iopub.execute_input":"2024-04-08T13:31:20.218907Z","iopub.status.idle":"2024-04-08T13:31:20.241178Z","shell.execute_reply.started":"2024-04-08T13:31:20.218868Z","shell.execute_reply":"2024-04-08T13:31:20.239822Z"}}
df2.isna().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.242627Z","iopub.execute_input":"2024-04-08T13:31:20.243019Z","iopub.status.idle":"2024-04-08T13:31:20.286486Z","shell.execute_reply.started":"2024-04-08T13:31:20.242987Z","shell.execute_reply":"2024-04-08T13:31:20.285529Z"}}
df3 = df2.dropna()
df3 = df3[df3["23. How old are you?"] < 100]
df3 = df3.loc[df3['1. What is your biggest reason for learning to code?'].isin(q1columns)]
df3.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.287636Z","iopub.execute_input":"2024-04-08T13:31:20.288319Z","iopub.status.idle":"2024-04-08T13:31:20.617873Z","shell.execute_reply.started":"2024-04-08T13:31:20.288287Z","shell.execute_reply":"2024-04-08T13:31:20.616689Z"}}
df3.boxplot(column= '23. How old are you?')

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.619481Z","iopub.execute_input":"2024-04-08T13:31:20.620605Z","iopub.status.idle":"2024-04-08T13:31:20.635403Z","shell.execute_reply.started":"2024-04-08T13:31:20.620564Z","shell.execute_reply":"2024-04-08T13:31:20.634108Z"}}
df3['24. Which of the following best represents how you think of yourself?'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.637474Z","iopub.execute_input":"2024-04-08T13:31:20.637988Z","iopub.status.idle":"2024-04-08T13:31:20.652557Z","shell.execute_reply.started":"2024-04-08T13:31:20.637947Z","shell.execute_reply":"2024-04-08T13:31:20.651308Z"}}
df3['8. About how many months have you been programming?'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.654339Z","iopub.execute_input":"2024-04-08T13:31:20.655038Z","iopub.status.idle":"2024-04-08T13:31:20.670382Z","shell.execute_reply.started":"2024-04-08T13:31:20.655001Z","shell.execute_reply":"2024-04-08T13:31:20.668959Z"}}
df3 = df3[df3['24. Which of the following best represents how you think of yourself?'].isin(gend)]
df3 = df3[df3['8. About how many months have you been programming?'].isin(month)]
df3['8. About how many months have you been programming?'] = df3['8. About how many months have you been programming?'].astype(float)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.675973Z","iopub.execute_input":"2024-04-08T13:31:20.676357Z","iopub.status.idle":"2024-04-08T13:31:20.685205Z","shell.execute_reply.started":"2024-04-08T13:31:20.676328Z","shell.execute_reply":"2024-04-08T13:31:20.683881Z"}}
df3['32. What is the highest degree or level of school you have completed?'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.687097Z","iopub.execute_input":"2024-04-08T13:31:20.687728Z","iopub.status.idle":"2024-04-08T13:31:20.871009Z","shell.execute_reply.started":"2024-04-08T13:31:20.687625Z","shell.execute_reply":"2024-04-08T13:31:20.869388Z"}}
df3 = df3.replace('Under $1,000', 0)
df3 = df3.replace('$1,000 to $2,999', 1999)
df3 = df3.replace('$40,000 to $49,999', 44999)
df3 = df3.replace('$75,000 to $89,999', 82499)
df3 = df3.replace('$60,000 to $74,999', 67499)
df3 = df3.replace('$10,000 to $14,999', 12499)
df3 = df3.replace('$20,000 to $24,999', 22499)
df3 = df3.replace('$25,000 to $29,999', 27499)
df3 = df3.replace('$30,000 to $34,999', 32499)
df3 = df3.replace('$3,000 to $4,999', 3999)
df3 = df3.replace('$15,000 to $19,999', 17499)
df3 = df3.replace('$35,000 to $39,999', 37499)
df3 = df3.replace('$7,000 to $9,999', 8499)
df3 = df3.replace('$90,000 to $119,999', 104999)
df3 = df3.replace('$5,000 to $6,999', 5999)
df3 = df3.replace('$50,000 to $59,999', 54999)
df3 = df3.replace('$120,000 to $159,999', 139999)
df3 = df3.replace("I don't want to answer", 0)
df3 = df3.replace("I don’t know", 0)
df3 = df3.replace("$200,000 to $249,999", 224999)
df3 = df3.replace("$250,000 or over", 250000)
df3 = df3.replace("$160,000 to $199,999", 179999)
df3 = df3[df3["22. About how much money did you earn last year from any job or employment (in US Dollars)? "] < 224999]
df3.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.872816Z","iopub.execute_input":"2024-04-08T13:31:20.873596Z","iopub.status.idle":"2024-04-08T13:31:20.898545Z","shell.execute_reply.started":"2024-04-08T13:31:20.873558Z","shell.execute_reply":"2024-04-08T13:31:20.896910Z"}}
encoder = OrdinalEncoder(categories = [gend])
df3["24. Which of the following best represents how you think of yourself?"] = encoder.fit_transform(df3[["24. Which of the following best represents how you think of yourself?"]])
encoder = OrdinalEncoder(categories = [q1columns])
df3["1. What is your biggest reason for learning to code?"] = encoder.fit_transform(df3[["1. What is your biggest reason for learning to code?"]])
encoder = OrdinalEncoder(categories = [education])
df3["32. What is the highest degree or level of school you have completed?"] = encoder.fit_transform(df3[["32. What is the highest degree or level of school you have completed?"]])

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:20.900689Z","iopub.execute_input":"2024-04-08T13:31:20.901081Z","iopub.status.idle":"2024-04-08T13:31:21.046581Z","shell.execute_reply.started":"2024-04-08T13:31:20.901048Z","shell.execute_reply":"2024-04-08T13:31:21.045428Z"}}
X = df3.iloc[: ,[0,1,2,3,4,5,6]]
km = KMeans(n_clusters = 4 ,init = 'random',n_init =10 ,max_iter =500 ,random_state =0)
y = km.fit_predict(X)
df_cluster = df3.iloc [: ,0:7]
df_cluster ['ClusterLabel']=y
df_cluster.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:21.048117Z","iopub.execute_input":"2024-04-08T13:31:21.048791Z","iopub.status.idle":"2024-04-08T13:31:22.463992Z","shell.execute_reply.started":"2024-04-08T13:31:21.048756Z","shell.execute_reply":"2024-04-08T13:31:22.462704Z"}}
a = sns.scatterplot( data = df_cluster , x="24. Which of the following best represents how you think of yourself?", y="22. About how much money did you earn last year from any job or employment (in US Dollars)? ",hue="ClusterLabel")
#plt.ylim(10, 40)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:22.466154Z","iopub.execute_input":"2024-04-08T13:31:22.467090Z","iopub.status.idle":"2024-04-08T13:31:22.479460Z","shell.execute_reply.started":"2024-04-08T13:31:22.467042Z","shell.execute_reply":"2024-04-08T13:31:22.478509Z"}}
df_cluster['ClusterLabel'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:22.481640Z","iopub.execute_input":"2024-04-08T13:31:22.482837Z","iopub.status.idle":"2024-04-08T13:31:29.410096Z","shell.execute_reply.started":"2024-04-08T13:31:22.482764Z","shell.execute_reply":"2024-04-08T13:31:29.408837Z"}}
model = KMeans()
visualizer = KElbowVisualizer (model , k =(2 ,10))
visualizer.fit(X)
visualizer.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:29.411593Z","iopub.execute_input":"2024-04-08T13:31:29.411943Z","iopub.status.idle":"2024-04-08T13:31:33.025632Z","shell.execute_reply.started":"2024-04-08T13:31:29.411914Z","shell.execute_reply":"2024-04-08T13:31:33.024349Z"}}
k=4
km = KMeans(k, random_state =1)
visualizer = SilhouetteVisualizer (km , colors ='yellowbrick')
visualizer.fit(X)
visualizer.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:33.027329Z","iopub.execute_input":"2024-04-08T13:31:33.027690Z","iopub.status.idle":"2024-04-08T13:31:33.507059Z","shell.execute_reply.started":"2024-04-08T13:31:33.027661Z","shell.execute_reply":"2024-04-08T13:31:33.505710Z"}}
sns.barplot( data = df_cluster , x="ClusterLabel", y="23. How old are you?")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:33.508617Z","iopub.execute_input":"2024-04-08T13:31:33.509002Z","iopub.status.idle":"2024-04-08T13:31:33.531811Z","shell.execute_reply.started":"2024-04-08T13:31:33.508969Z","shell.execute_reply":"2024-04-08T13:31:33.530319Z"}}
x = df_cluster.iloc[:,[0,1,2,4,5,6]]
y = df_cluster.iloc[:,7]
x.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:33.533423Z","iopub.execute_input":"2024-04-08T13:31:33.533854Z","iopub.status.idle":"2024-04-08T13:31:33.552190Z","shell.execute_reply.started":"2024-04-08T13:31:33.533817Z","shell.execute_reply":"2024-04-08T13:31:33.550753Z"}}
X_train ,X_test , y_train , y_test = model_selection.train_test_split(x,y, test_size =0.5,random_state = 7)
scaler = MinMaxScaler ()
scaler .fit ( X_train )
X_train = scaler . transform ( X_train )
X_test = scaler . transform ( X_test )

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:33.554160Z","iopub.execute_input":"2024-04-08T13:31:33.554653Z","iopub.status.idle":"2024-04-08T13:31:33.572893Z","shell.execute_reply.started":"2024-04-08T13:31:33.554610Z","shell.execute_reply":"2024-04-08T13:31:33.571734Z"}}
knn = KNeighborsClassifier( n_neighbors = 8, weights ="uniform", metric ="euclidean")
knn.fit(X_train , y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:33.574750Z","iopub.execute_input":"2024-04-08T13:31:33.575175Z","iopub.status.idle":"2024-04-08T13:31:34.083620Z","shell.execute_reply.started":"2024-04-08T13:31:33.575136Z","shell.execute_reply":"2024-04-08T13:31:34.081325Z"}}
y_knn_predict = knn.predict( X_test )
accuracy = accuracy_score (y_test , y_knn_predict )
accuracy

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.084591Z","iopub.status.idle":"2024-04-08T13:31:34.085005Z","shell.execute_reply.started":"2024-04-08T13:31:34.084812Z","shell.execute_reply":"2024-04-08T13:31:34.084829Z"}}
model1 = DecisionTreeClassifier ()
model2 = KNeighborsClassifier ()
model3 = LogisticRegression (max_iter =10000)
clf_ensemble = VotingClassifier(estimators =[( 'dt', model1 ) , ('knn', model2 ) , ('lr',model3 ) ] , voting = 'soft')
clf_ensemble = clf_ensemble.fit( X_train , y_train )

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.086083Z","iopub.status.idle":"2024-04-08T13:31:34.086486Z","shell.execute_reply.started":"2024-04-08T13:31:34.086285Z","shell.execute_reply":"2024-04-08T13:31:34.086301Z"}}
y_ens_pred = clf_ensemble . predict ( X_train )
y_enstest_pred = clf_ensemble . predict ( X_test )
train_score = accuracy_score( y_train , y_ens_pred )
test_score = accuracy_score (y_test , y_enstest_pred )
print(train_score,test_score)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.087417Z","iopub.status.idle":"2024-04-08T13:31:34.087813Z","shell.execute_reply.started":"2024-04-08T13:31:34.087622Z","shell.execute_reply":"2024-04-08T13:31:34.087638Z"}}
clf_forest = RandomForestClassifier (n_estimators =13 , random_state =1)
clf_forest = clf_forest.fit(X_train , y_train)
y_train_pred = clf_forest.predict (X_train)
y_test_pred = clf_forest.predict (X_test)
train_score = accuracy_score ( y_train , y_train_pred )
test_score = accuracy_score (y_test ,y_test_pred )
print ('train / test accuracies',(train_score , test_score ))


# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.088821Z","iopub.status.idle":"2024-04-08T13:31:34.089180Z","shell.execute_reply.started":"2024-04-08T13:31:34.089000Z","shell.execute_reply":"2024-04-08T13:31:34.089015Z"}}
cm = confusion_matrix(y_test, y_enstest_pred, labels=clf_ensemble.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_ensemble.classes_)
disp.plot()
disp.ax_.set_title("Voting Classifier")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:53.831179Z","iopub.execute_input":"2024-04-08T13:31:53.831920Z","iopub.status.idle":"2024-04-08T13:31:53.855626Z","shell.execute_reply.started":"2024-04-08T13:31:53.831880Z","shell.execute_reply":"2024-04-08T13:31:53.854682Z"}}
X_train , X_test , y_train , y_test = model_selection . train_test_split (x, y, test_size = 0.3)
clf_tree = DecisionTreeClassifier( criterion ='gini', max_depth =4)
clf_tree = clf_tree.fit( X_train , y_train )
y_pred_tree = clf_tree . predict ( X_test )

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:33:41.079740Z","iopub.execute_input":"2024-04-08T13:33:41.080143Z","iopub.status.idle":"2024-04-08T13:33:41.208555Z","shell.execute_reply.started":"2024-04-08T13:33:41.080113Z","shell.execute_reply":"2024-04-08T13:33:41.207182Z"}}
X_train ,X_test , y_train , y_test = train_test_split (x,y, test_size =0.3 , random_state =4)
scaler = MinMaxScaler ()
scaler .fit ( X_train )
X_train = scaler . transform ( X_train )
X_test = scaler . transform ( X_test )
clf_log = LogisticRegression ()
clf_log = clf_log.fit( X_train , y_train )
clf_log. get_params ()
y_pred_log = clf_log . predict ( X_test )


# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:34:19.048719Z","iopub.execute_input":"2024-04-08T13:34:19.053246Z","iopub.status.idle":"2024-04-08T13:34:19.548836Z","shell.execute_reply.started":"2024-04-08T13:34:19.053184Z","shell.execute_reply":"2024-04-08T13:34:19.547543Z"}}
cm = confusion_matrix(y_test, y_pred_log, labels=clf_log.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_log.classes_)
disp.plot()
disp.ax_.set_title("Logistic Regression")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.096569Z","iopub.status.idle":"2024-04-08T13:31:34.097378Z","shell.execute_reply.started":"2024-04-08T13:31:34.097163Z","shell.execute_reply":"2024-04-08T13:31:34.097183Z"}}
Accuracy = metrics.accuracy_score(y_test , y_test_pred)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.100614Z","iopub.status.idle":"2024-04-08T13:31:34.101033Z","shell.execute_reply.started":"2024-04-08T13:31:34.100836Z","shell.execute_reply":"2024-04-08T13:31:34.100854Z"}}
Sensitivity = metrics.recall_score(y_test , y_test_pred, average = "weighted")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.102217Z","iopub.status.idle":"2024-04-08T13:31:34.103060Z","shell.execute_reply.started":"2024-04-08T13:31:34.102849Z","shell.execute_reply":"2024-04-08T13:31:34.102868Z"}}
y_score = clf_forest.fit(X_train, y_train).predict_proba(X_test)
auc = metrics.roc_auc_score(
    y_test,
    y_score,
    multi_class="ovr",
    average="micro",
)

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.104371Z","iopub.status.idle":"2024-04-08T13:31:34.105186Z","shell.execute_reply.started":"2024-04-08T13:31:34.104973Z","shell.execute_reply":"2024-04-08T13:31:34.104992Z"}}
Specificity = specificity_score(y_test, y_test_pred, average = "weighted")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.106529Z","iopub.status.idle":"2024-04-08T13:31:34.107307Z","shell.execute_reply.started":"2024-04-08T13:31:34.107085Z","shell.execute_reply":"2024-04-08T13:31:34.107105Z"}}
F1_score = metrics.f1_score(y_test, y_test_pred, average = "weighted")

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.108781Z","iopub.status.idle":"2024-04-08T13:31:34.109180Z","shell.execute_reply.started":"2024-04-08T13:31:34.108987Z","shell.execute_reply":"2024-04-08T13:31:34.109003Z"}}
print({"Accuracy":Accuracy,"Sensitivity":Sensitivity,"Specificity":Specificity,"F1_score":F1_score})

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.110336Z","iopub.status.idle":"2024-04-08T13:31:34.110782Z","shell.execute_reply.started":"2024-04-08T13:31:34.110579Z","shell.execute_reply":"2024-04-08T13:31:34.110597Z"}}
print(classification_report (y_test , y_pred_tree))

# %% [code] {"execution":{"iopub.status.busy":"2024-04-08T13:31:34.112275Z","iopub.status.idle":"2024-04-08T13:31:34.113009Z","shell.execute_reply.started":"2024-04-08T13:31:34.112800Z","shell.execute_reply":"2024-04-08T13:31:34.112819Z"}}
X_train ,X_test , y_train , y_test = model_selection .train_test_split (x,y, test_size =0.5 ,random_state =7)
X_train_new , X_val , y_train_new , y_val = model_selection . train_test_split ( X_train , y_train , test_size =0.1 , random_state =4)
max_depth_range = range (1 , 16)
val_results = []
train_results = []
for k in max_depth_range :
    clf_2 = RandomForestClassifier (n_estimators = k , random_state =1)
    clf_2 = clf_2 .fit( X_train_new , y_train_new )
    pred_train_new = clf_2 . predict ( X_train_new )
    train_score = metrics . accuracy_score (y_train_new , pred_train_new )
    train_results . append ( train_score )
    pred_val = clf_2 . predict ( X_val )
    val_score = metrics . accuracy_score (y_val , pred_val )
    val_results . append ( val_score )
    
plt. plot ( max_depth_range , val_results , 'g-',label ='Val score')
plt. plot ( max_depth_range , train_results , 'r-', label ='Train score')
plt. ylabel ('Score')
plt. xlabel ('Model complexity : tree depth')
plt. legend ()
plt. show ()
