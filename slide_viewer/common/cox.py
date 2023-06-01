import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize

train_table = pd.read_excel('E:/Code/HistoSlider/test-data/life_line_max_train.xlsx')
test_table = pd.read_excel('E:/Code/HistoSlider/test-data/life_line_max_test.xlsx')

for c, i in enumerate(test_table['histological_type']):
    if not 'Colon' in i:
        test_table = test_table.drop(c, axis=0)

train_os = train_table[['OS_time', 'OS']]
test_os = test_table[['OS_time', 'OS']]

train_feature = train_table.iloc[:, 9:]
test_feature = test_table.iloc[:, 8:]

train_T = train_feature['OS_time']
train_E = train_feature['OS']
test_T = test_feature['OS_time']
test_E = test_feature['OS']

cls_list = []
for i in train_feature.columns[2:]:
    cph = CoxPHFitter()
    cph.fit(pd.concat([train_os, train_feature[i]], axis=1), duration_col='OS_time', event_col='OS')
    if float(cph.summary['p']) <= 0.5:
        cls_list.append(i)

select_table = train_table[cls_list]
select_test_table = test_table[cls_list]
final_train = pd.concat([train_os, select_table], axis=1)
final_test = pd.concat([test_os, select_test_table], axis=1)

for col in final_train.columns:
    if 'BACK' in col:
        final_train = final_train.drop(col, axis=1)

for col in final_train.columns:
    if 'MHLS' in col:
        final_train = final_train.drop(col, axis=1)


for col in final_test.columns:
    if 'BACK' in col:
        final_test = final_test.drop(col, axis=1)

for col in final_test.columns:
    if 'MHLS' in col:
        final_test = final_test.drop(col, axis=1)
#


# f, aa = plt.subplots(2, 1)
cph = CoxPHFitter()
cph.fit(final_train, duration_col='OS_time', event_col='OS', show_progress=True)
cph.print_summary()
# aa[0] = cph.plot()
# aa[1] = cph.plot()
cph.plot()
plt.savefig('./cox_train_res.jpg', bbox_inches='tight')
plt.show()
plt.close()

cph1 = CoxPHFitter()
cph1.fit(final_test, duration_col='OS_time', event_col='OS', show_progress=True)
cph1.print_summary()
cph1.plot()
plt.savefig('./cox_test_res.jpg', bbox_inches='tight')
plt.show()
plt.close()





# fig, ax = plt.subplots(1, 2)

train_hr_ratio = cph.predict_partial_hazard(final_train)
kmf = KaplanMeierFitter()
train_flag = (train_hr_ratio <= 1)
ax = plt.subplot(111)
kmf.fit(train_T[train_flag], event_observed=train_E[train_flag], label="low")
kmf.plot_survival_function(ax=ax)
kmf.fit(train_T[~train_flag], event_observed=train_E[~train_flag], label="high")
kmf.plot_survival_function(ax=ax)
a = plt.savefig('./km_train_res.jpg', bbox_inches='tight')
print(a)
plt.show()

plt.close()

test_hr_ratio = cph.predict_partial_hazard(select_test_table)
test_flag = (test_hr_ratio <= 0.4)
ax = plt.subplot(111)
kmf.fit(test_T[test_flag], event_observed=test_E[test_flag], label="low")
kmf.plot_survival_function(ax=ax)
kmf.fit(test_T[~test_flag], event_observed=test_E[~test_flag], label="high")
kmf.plot_survival_function(ax=ax)
plt.savefig('./km_test_res.jpg', bbox_inches='tight')
plt.show()

plt.close()








