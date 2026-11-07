from load_data import NHANES, FEATURE_BLOCKS, TARGET
from statsmodels.stats.proportion import proportion_confint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

nh = NHANES()
train, test = nh.get_nhanes(force_rebuild=False)


bins = [18, 40, 60, 80]
labels = ['18-39', '40-59', '60+']

# prevalence is the proportion of positive instances (or a target class) in a dataset (N) --> Formula: Actual Positive / Total N
def prevalence_table(df, group_col):
    t = df.groupby(group_col, observed=True)[TARGET].agg(
        n='size', positives='sum', prevalence='mean'
    ).reset_index()

    lo, hi = proportion_confint(t['positives'], t['n'], method='wilson')
    t['ci_low'], t['ci_high'] = lo, hi
    return t

for d in (train, test):
    d['age_band'] = pd.cut(d['age'], bins=bins, labels=labels, include_lowest=True)

print("\nPrevalence by age_band")
t_train = prevalence_table(train, 'age_band')
t_test = prevalence_table(test, 'age_band')
print(t_train , '\n')
print(t_test)

lower_err = t_train['prevalence'] - t_train['ci_low']
upper_err = t_train['ci_high'] - t_train['prevalence']
asymmetric_error = np.array([lower_err, upper_err])

sns.set_theme(style='whitegrid')
plt.figure(figsize=(8,5))
ax = sns.barplot(
    data=t_train,
    x='age_band',
    y='prevalence',
    hue='age_band',
    palette='muted',
    legend=False
)

plt.errorbar(
    x=range(len(t_train)),
    y=t_train['prevalence'],
    yerr=asymmetric_error,
    fmt='none',
    c='black',
    capsize=6,
    elinewidth=2
)

plt.title('Prevalence by Age Band (Train Set) with 95% Wilson CI')
plt.xlabel('Age Group')
plt.ylabel('Prevalence Rate')
plt.ylim(0, max(t_train['ci_high']) * 1.15)
plt.show()

print("\nPrevalence by sex")
print(prevalence_table(train, 'sex') , '\n')
print(prevalence_table(test, 'sex'))

print("\nPrevalence by is_asian_descent")
print(prevalence_table(train, 'is_asian_descent') , '\n')
print(prevalence_table(test, 'is_asian_descent'))

