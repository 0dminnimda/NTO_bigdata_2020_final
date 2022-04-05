
# load
import pandas as pd
# import lightgbm

data = pd.read_csv("X_train.csv", index_col=0)
data["mark"] = pd.read_csv("y_train.csv", index_col=0)["mark"]

stud_info = pd.read_csv("studs_info.csv", index_col=False)

X_validation = pd.read_csv("X_test.csv", index_col=0)
# rename columns
field_map = {
    "STD_ID": "stud",
    "НАПРАВЛЕНИЕ": "profession",
    "ГОД": "year",
    "АТТЕСТАЦИЯ": "exam_type",
    "ДИСЦИПЛИНА": "discipline",
    "КУРС": "course",
    "СЕМЕСТР": "semester",

    "   number": "number",
    "Пол": "sex",
    "Статус": "state",
    "Дата выпуска": "release_date",
    "Категория обучения": "category",
    "Форма обучения": "study_kind",
    "Шифр": "cipher",
    "направление (специальность)": "speciality",
    "   ": "what?",
    "Образование": "lvl_of_education",
    "Дата выдачи": "issue_date",
    "Что именно закончил": "education",
}

data.rename(columns=field_map, inplace=True)
X_validation.rename(columns=field_map, inplace=True)
stud_info.rename(columns=field_map, inplace=True)
stud_info.drop(stud_info[stud_info["stud"] == 92222].index, inplace=True)

# stud_info[np.isin(stud_info["number"], range(850, 900))].sort_values(by=["stud"])
# all(stud_info.groupby("speciality")["cipher"].nunique().eq(1))# and
all(stud_info.groupby("cipher")["speciality"].nunique().eq(1))
g = stud_info.groupby("speciality")["cipher"].nunique()
print(g[g != 1])

set(stud_info[stud_info["speciality"] == "Журналистика"]["cipher"])
# 203283
# remove duplicate entries (older ones)
stud_info = stud_info.sort_values(by=["stud", "issue_date"], na_position="first")
stud_info.drop_duplicates(subset=["stud"], keep="last", inplace=True)

import numpy as np
assert len(stud_info[np.isin(stud_info["stud"], stud_info[stud_info.duplicated(subset=["stud"])])]) == 0

# clean up

# for each stud: year == course + const
# for each stud: course == ceil(semester / 2)
# therefore they are noise
fields = ["year", "course"]
data.drop(fields, axis=1, inplace=True)
X_validation.drop(fields, axis=1, inplace=True)

# all nulls and not present in data / validation
stud_info.drop(stud_info[stud_info["stud"] == 92222].index, inplace=True)

# for each stud: all number_s are equal
assert all(stud_info.groupby("number")["stud"].nunique().le(1)) and all(stud_info.groupby("stud")["number"].nunique().le(1))
fields = ["number", "issue_date", "release_date"]
stud_info.drop(fields, axis=1, inplace=True)
{
# ('НС', 'СР'): 4,
#  ('ОСН', 'СР'): 3,
#  ('НС', 'СП'): 5,
#  ('СР', 'СП'): 111,
#  ('ОСН', 'СП'): 24,
#  ('ОО', 'СР'): 22,
#  ('ОО', 'СП'): 131,
 ('НП', 'СР'): 1,
 ('НП', 'СП'): 10,
 ('СП', 'СП'): 7,
 ('СР', 'СР', 'СП'): 1,
 ('СР', 'СР'): 1,
 ('СП', 'СР'): 1,
 ('СП', 'НП'): 1}

# ('ОО', 'СР'      )
# (      'СР', 'СП')
# ('ОО',       'СП')

# ('ОО',        'СР', 'СП')
# (      'ОСН', 'СР'      )
# (      'ОСН', 'СП'      )

('ОО', 'ОСН', 'СР', 'СП')
('НС',        'СР'      )
('НС',              'СП')

# # SeriesGroupBy.cummax()
stud_info
stud_info.fillna({"lvl_of_education": "НЕТ", "what?": 0.0}, inplace=True)

data = data.merge(stud_info, how="left", on="stud")
X_validation = X_validation.merge(stud_info, how="left", on="stud")
data

# encode labels
from sklearn import preprocessing

fields = ["discipline", "profession", "exam_type", "sex", "category", "speciality", "education", "state", "cipher"]
le_s = {
    field_name: preprocessing.LabelEncoder().fit(pd.concat([data[field_name], X_validation[field_name]]))
    for field_name in fields}

order = [
    "НЕТ",  # 190  Нет данных
    "ОО",   # 160  Начальное общее образование
    "ОСН",  # 32   Основное общее образование
    "НС",   # 14   Неполное среднее образование
    "СР",   # 4101 Среднее общее образование
    "НВ",   # 2    Неполное высшее образование
    "НП",   # 50   Начальное/Незаконченное? профессиональное образование
    "СП",   # 916  Среднее профессиональное образование
]
le_s["lvl_of_education"] = preprocessing.LabelEncoder().fit(order)

order = ["В", "Д", "З"]  # вечернее, дневное, заочное
le_s["study_kind"] = preprocessing.LabelEncoder().fit(order)

for field_name, le in le_s.items():
    data[field_name] = le.transform(data[field_name])
    X_validation[field_name] = le.transform(X_validation[field_name])

# 69.0 to 69
fields = ["semester", "what?"]
for field_name in fields:
    data[field_name] = data[field_name].astype(int)
    X_validation[field_name] = X_validation[field_name].astype(int)

# normalize
data["semester"] -= 1
X_validation["semester"] -= 1
data
# means
fields = ["stud", "profession", "discipline", "speciality", "education", "cipher"]
for field_name in fields:
    mean_mark = data.groupby(field_name).mean()["mark"]

    mean_name = field_name + "_mean"
    data[mean_name] = data[field_name].map(mean_mark)
    X_validation[mean_name] = X_validation[field_name].map(mean_mark)

# create dummy variables
columns = []#"exam_type"]#, "discipline", "profession"]
data = pd.get_dummies(data, columns=columns)
X_validation = pd.get_dummies(X_validation, columns=columns)

# remove unneeded data
# use previous fields
# fields = ["stud", "discipline", "profession"]
data.drop(fields, axis=1, inplace=True)
X_validation.drop(fields, axis=1, inplace=True)

data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

marks = data.pop("mark")
X_train, X_test, y_train, y_test = train_test_split(data, marks, shuffle=True, test_size=0.2)

import autosklearn.classification
import autosklearn.metrics

from get_config import config

import time
automl = autosklearn.classification.AutoSklearnClassifier(metric=autosklearn.metrics.mean_absolute_error, logging_config=config)
print("start", time.time())
automl.fit(X_train, y_train)
print(time.time())


def p():
    print(automl.cv_results_)
    print(automl)
    print({name: getattr(automl, name) for name in dir(automl)})


p()

# parameters = {
#     # LGBMRegressor(min_child_samples=1, min_child_weight=1.0, n_estimators=1000,
#     #            num_leaves=50, random_state=42, reg_alpha=1.0, reg_lambda=1.0))

#     'boosting_type': ('gbdt', 'dart', 'goss', 'rf',),
#     'num_leaves': [2, 15, 31],
#     "n_estimators": np.linspace(1, 1000, 4, dtype=int),
#     # "min_split_gain": np.linspace(0, 1, 4),
#     # "min_child_weight": [1e-3, 1.],
#     # "min_child_samples": [1, 30],
#     "reg_alpha": np.linspace(0, 1, 4),
#     "reg_lambda": np.linspace(0, 1, 4),
# }
# parameters = {
#     'boosting_type': ('gbdt', 'dart', 'goss', 'rf',),
#     'num_leaves': [1, 50],
#     "n_estimators": [1, 1000],
#     "min_split_gain": [0., 1.],
#     "min_child_weight": [1e-3, 1.],
#     "min_child_samples": [1, 30],
#     "reg_alpha": [0., 1.],
#     "reg_lambda": [0., 1.],
# }

# clf = GridSearchCV(lightgbm.LGBMRegressor(random_state=42), parameters, verbose=3, cv=2, scoring='neg_mean_absolute_error')  # n_jobs=10,
# clf.fit(X_train, y_train)

# print(clf.cv_results_)

# print(clf.best_estimator_)

from sklearn.metrics import mean_absolute_error, r2_score

pred_mark = automl.predict(X_test)
print(mean_absolute_error(y_test, pred_mark))
print(r2_score(y_test, pred_mark))

print("gg", time.time())
automl.fit(data, marks)
print(time.time())

p()

y_pred = pd.read_csv("sample_submission.csv", index_col=0)
y_pred["mark"] = automl.predict(X_validation)

y_pred.to_csv("baseline_submission.csv")
#ya tyt bil
#I was here