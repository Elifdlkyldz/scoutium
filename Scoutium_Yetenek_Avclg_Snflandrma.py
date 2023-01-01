
#scoutium_attributes ;

#task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id : İlgili maçın id'si
#evaluator_id : Değerlendiricinin(scout'un) id'si
#player_id : İlgili oyuncunun id'si
#position_id : İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#1: Kaleci
#2: Stoper
#3: Sağ bek
#4: Sol bek
#5: Defansif orta saha
#6: Merkez orta saha
#7: Sağ kanat
#8: Sol kanat
#9: Ofansif orta saha
#10: Forvet
#analysis_id : Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
#attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
#attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

#scoutium_potential_labels;

#task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
#match_id : İlgili maçın id'si
#evaluator_id : Değerlendiricinin(scout'un) id'si
#player_id : İlgili oyuncunun id'si
#potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import RobustScaler , StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from matplotlib import rc,rcParams
import itertools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


## scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutuyoruz.


df1 = pd.read_csv("datasets/scoutium_attributes.csv", sep= ';') #direk noktalı virgülde eklenebilir. / seperatior/ön tanımlı değer virgül.



df2= pd.read_csv("datasets/scoutium_potential_labels.csv", sep= ';')

df1.head()
df2.head()




## Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriyoruz.
#("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)


df = pd.merge(df1, df2, how='left', on=['task_response_id','match_id', 'evaluator_id', "player_id" ]) 
#/fazladan değ. türemesin diye.4ü üzerinden/index sayısı artmasın diye.

def check_df(dataframe, head=5): ##veri setini tanıma fonk.
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırıyoruz.

df = df[df.position_id != 1]     
# Remove the Keeper (1) class in position / df[~df.position_id == 1] / or df.drop(df[df['position_id'] == 1].index, inplace = True)

df[["position_id"]].value_counts()



##potential_label içerisindeki below_average sınıfını veri setinden kaldırıyoruz.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df = df[df['potential_label'] != 'below_average']

df[["potential_label"]].value_counts()


#potential_labels grafiklendirme;

f,ax=plt.subplots(1,2,figsize=(10,5))
df["potential_label"].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Distribution')
ax[0].set_ylabel('')
sns.countplot(df["potential_label"],ax=ax[1])
ax[1].set_title('Count')
plt.show()


df.groupby(df=['potential_label']).aggregate({'attribute_values' : 'mean' })

## Oluşturduğumuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturuyoruz. Bu pivot table'da her satırda bir oyuncu
#olacak şekilde manipülasyon yapıyoruz.
#*İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#“attribute_value” olacak şekilde pivot table’ı oluşturuyoruz.

df_pivot = pd.pivot_table(df, values='attribute_value' ,
                          index=['player_id', 'position_id', 'potential_label'],
                          columns='attribute_id')



#“reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayoruz ve “attribute_id” sütunlarının isimlerini stringe çeviriyoruz.


df_pivot = df_pivot.reset_index(inplace=True)

df_pivot.columns = df_pivot.columns.map(str)
df_pivot.head()
df_pivot.columns
df_pivot.dtypes



## Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediyoruz.

#label enc. kategorik değişkenleri işlemek için.
# label encoding / binary encoding işlemini 2 sınıflı kategorik değişkenlere uyguluyoruz. bu iki sınıfı 1-0 şeklinde encodelamış oluyoruz.
# one-hot encoder ise ordinal sınıflı kategorik değişkenler için uyguluyoruz. sınıfları arasında fark olan
# değişkenleri sınıf sayısınca numaralandırıp kategorik değişken olarak df e gönderiyor.



le = LabelEncoder()

df_pivot["potential_label"] = le.fit_transform(df_pivot["potential_label"])

df_pivot.head()

#num_cols = df_pivot. columns[3:] #virgülden sonraki ilk 3

##Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayoruz.

num_cols = [col for col in df_pivot.columns if col not in ["player_id", "position_id", "potential_label"]]
df_pivot_num = df_pivot[num_cols]
df_pivot_num.shape


##Korelasyona bakıyoruz;
##################################

df_pivot[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_pivot[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()



## Kaydettiğimiz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için Standard Scaler uyguluyoruz.


StandardScaler().fit_transform(df_pivot[num_cols])

#Standartlaştırma
#Değişken (özellik) sütunlarının ortalama değeri 0 ve standart sapması 1 olacak şekilde standart normal dağılım oluşturmaktır.





## Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
#geliştiriyoruz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırıyoruz.)


y = df_pivot["potential_label"] #Y BAĞIMLI, X BAĞIMSIZ
X = df_pivot.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

df_pivot.head()

# Train verisi ile model kurup, model başarısını değerlendiriyoruz.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


f,ax=plt.subplots(1,2,figsize=(10,5))
y_train.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Distribution')
ax[0].set_ylabel('')
sns.countplot(y_train,ax=ax[1])
ax[1].set_title('Count')
plt.show()



def base_models(X, y, scoring):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for i in scores:
    base_models(X, y, i)

## Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriyoruz.


def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)

# Stacking & Ensemble Learning

def voting_classifier(best_models, X, y):
    print("Voting Classifier...") #işlemşn başladığını görmek için yazdık bunu.

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), #best modelsden knn gelicek.bunları bşz seçtik. bestde çıkan sonuçlarda en iyileri seçebilirisn!!
                                              ('RF', best_models["RF"]), #bestden rf gelicek.
                                              ('LightGBM', best_models["LightGBM"])], #aynı işlem.
                                  voting='soft').fit(X, y) #hepsini fit edicek. voting-clf nin cv hatasına bakıcaz.

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


X.columns
random_user = X.sample(1, random_state=45) #rastgele bi kullanıcı seçtik.
voting_clf.predict(random_user) #bu kullanıcı için diabet tahmini gerçekleştirdik.

joblib.dump(voting_clf, "voting_clf2.pkl") #bu modeli kayıt edicez. isim bana kalmış.

new_model = joblib.load("voting_clf2.pkl") #modeli kayıt ettim ve load diyerek dosyamı yükledim.
new_model.predict(random_user)
