import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.impute import SimpleImputer

'''
Read data
G is gender 
A is anchor age 
B is obeze (ICD code 27801) 
In is the number of times of ICU stays 
It is the average time of ICU stays 
Ic is CCU stays (careunit is Coronary Care Unit)
'''

file_icd = "csv/diagnoses_icd.csv"
file_AG = "csv/patients.csv"
file_I1 = "csv/icustays.csv"
file_I2 = "csv/transfers.csv"

# Read CSV
data_icd = pd.read_csv(file_icd)
data_AG = pd.read_csv(file_AG)
data_I1 = pd.read_csv(file_I1)
data_I2 = pd.read_csv(file_I2)

# Filter ICD
icd = ['40201', '40291', '40401', '40403', '40411', '40413', '40491', '40493']

# Select columns that meet the specified ICD
filtered = data_icd[data_icd['icd_code'].isin(icd)]

# Filter all subject_id
subject_id = filtered['subject_id']

# Filter all subject_id's G
G_arr = data_AG[data_AG['subject_id'].isin(subject_id)]['gender']

# Filter all subject_id's A
A_arr = data_AG[data_AG['subject_id'].isin(subject_id)]['anchor_age']

# Filter all subject_id's B
B_arr = data_icd[(data_icd['subject_id'].isin(subject_id)) & (data_icd['icd_code'] == '27801')]

# Filter all subject_id's In
In_arr = data_I1.groupby('subject_id').size()

# Filter all subject_id's It
data_I1['stay_length'] = (pd.to_datetime(data_I1['outtime']) - pd.to_datetime(data_I1['intime'])).dt.total_seconds()
icu_stays = data_I1[data_I1['subject_id'].isin(subject_id)]
It_arr = icu_stays.groupby('subject_id')['stay_length'].mean()

# Filter all subject_id's Ic
data_I2['careunit'] = data_I2['careunit'].astype(str)
Ic_arr = data_I2.groupby('subject_id')['careunit'].apply(lambda x: ('Coronary Care Unit (CCU)' in x.values))

'''
Assign values
'''

# Assign values to G
G_arr = G_arr.map({'M': 2.6, 'F': 2})

# Assign values to A
bins = [0, 15, 35, 60, np.inf]
labels = [1, 3, 10, 50]
A_arr = pd.cut(A_arr, bins=bins, labels=labels, include_lowest=True)

# Assign values to B
B_arr = B_arr['subject_id'].apply(lambda x: 3 if x in B_arr['subject_id'].values else 1)

# Assign values to In
In_arr /= 5

# Assign values to It
It_arr = pd.cut(It_arr, bins=[0, 60000, 180000, np.inf], labels=[1, 3, 25], include_lowest=True)

# Assign values to Ic
Ic_arr = Ic_arr.map({True: 3, False: 1})

# Fill missing values with median
imputer = SimpleImputer(strategy='median')

G_arr_filled = imputer.fit_transform(G_arr.values.reshape(-1, 1))
G_arr = pd.Series(G_arr_filled.flatten())

A_arr_filled = imputer.fit_transform(A_arr.values.reshape(-1, 1))
A_arr = pd.Series(A_arr_filled.flatten())

B_arr_filled = imputer.fit_transform(B_arr.values.reshape(-1, 1))
B_arr = pd.Series(B_arr_filled.flatten())

In_arr_filled = imputer.fit_transform(In_arr.values.reshape(-1, 1))
In_arr = pd.Series(In_arr_filled.flatten())

It_arr_filled = imputer.fit_transform(It_arr.values.reshape(-1, 1))
It_arr = pd.Series(It_arr_filled.flatten())

Ic_arr_filled = imputer.fit_transform(Ic_arr.values.reshape(-1, 1))
Ic_arr = pd.Series(Ic_arr_filled.flatten())

'''
Calculate Fp & Fi & f & m
Fp is the physiological factor, Fi is the ICU factor, f is the Readmission factor and m is the Mortality.
'''

f_arr = []
m_arr = []
for i in range(len(subject_id)):
    try:
        Fp = np.e ** 1.61 * G_arr.iloc[i] + np.e**(A_arr.iloc[i] / 10) + np.e ** 1.35 * B_arr.iloc[i]
        Fi = np.e ** (In_arr.iloc[i] + It_arr.iloc[i] / 5 + Ic_arr.iloc[i])
        f = (0.6 * Fp + 0.4 * Fi) / 2500 * 1.2 + np.random.uniform(-0.05, 0.05)
        if f > 100:
            f = 100
        f_arr.append(f)
        m = 1 / (1 + np.e**(-f))
        m_arr.append(m)
    except IndexError:
        pass

# Scale m to [0, 1] with random noise
m_arr = np.array(m_arr).reshape(-1, 1)
scaler = MinMaxScaler()
m_arr_scaled = scaler.fit_transform(m_arr)
m_arr_scaled = m_arr_scaled.flatten().tolist()
m_arr_scaled = [m + np.random.uniform(-0.1, 0.1) for m in m_arr_scaled]
m_arr_scaled = np.clip(m_arr_scaled, 0, 1).tolist()

'''
Calculate the descriptive statistics of m and plot the distribution of m
'''
# Calculate the descriptive statistics
mean = np.mean(m_arr_scaled)
median = np.median(m_arr_scaled)
variance = np.var(m_arr_scaled)

labels = ['Mean', 'Median', 'Variance']
values = [mean, median, variance]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values, color=['skyblue', 'orange', 'lightgreen'])
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.title('Descriptive Statistics of Mortality')
plt.ylabel('Value')
plt.show()

# Plot the distribution of m
plt.figure(figsize=(10, 6))
sns.histplot(m_arr_scaled, kde=True, color='skyblue', bins=30)

plt.title('Distribution of Mortality', fontsize=16)
plt.xlabel('Mortality', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()

'''
Plot the readmission factor & mortality of patients
'''

plt.figure()

plt.plot(f_arr, 'b', label='Readmission factor(F)')
plt.plot(m_arr_scaled, 'g', label='Mortality(M)')

plt.title('The readmission factor & mortality of patients')
plt.xlabel('Subject ID')
plt.ylabel('Value')

plt.legend()
plt.grid(True)
plt.show()

'''
Visualize f & d using a scatter plot
'''

plt.figure()

plt.scatter(range(len(f_arr)), f_arr, color='b', label='Readmission factor(F)', alpha=0.6)
plt.scatter(range(len(m_arr_scaled)), m_arr_scaled, color='g', label='Mortality(M)', alpha=0.6)

plt.title('Scatter plot of the readmission factor & mortality')
plt.xlabel('Subject ID')
plt.ylabel('Value')

plt.legend()
plt.grid(True)
plt.show()

'''
Calculate & plot the weights of variables (G, A, B, In, It, Ic)
'''

min_length = min(len(A_arr), len(In_arr), len(It_arr), len(Ic_arr))
weights_list = []

for i in range(min_length):
    weights = {
        'G': 1.61 * np.e ** 1.61,
        'A': 0.1 * np.e ** (A_arr.iloc[i] / 10),
        'B': 1.35 * np.e ** 1.35,
        'In': np.e ** In_arr.iloc[i],
        'It': 0.2 * np.e ** (It_arr.iloc[i] / 5),
        'Ic': np.e ** Ic_arr.iloc[i]
    }

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    weights_list.append(weights)

colors = ['b', 'g', 'r', 'c', 'm', 'y']
labels = ['G', 'A', 'B', 'In', 'It', 'Ic']

plt.figure(figsize=(10,7))
for i, weights in enumerate(weights_list):
    plt.bar(labels, weights.values(), color=colors, alpha=0.1)

# Plot the average weights
average_weights = {k: np.mean([weights[k] for weights in weights_list]) for k in labels}
plt.bar(average_weights.keys(), average_weights.values(), color=colors)

plt.title('Weights of Variables')
plt.xlabel('Variable')
plt.ylabel('Weight')
plt.grid(True)
plt.show()

'''
Select the best 5 features using SelectKBest
'''

lengths = [len(G_arr), len(A_arr), len(B_arr), len(In_arr), len(It_arr), len(Ic_arr), len(f_arr)]
min_length = min(lengths)

G_arr = G_arr[:min_length]
A_arr = A_arr[:min_length]
B_arr = B_arr[:min_length]
In_arr = In_arr[:min_length]
It_arr = It_arr[:min_length]
Ic_arr = Ic_arr[:min_length]
f_arr = f_arr[:min_length]

X = pd.concat([G_arr, A_arr, B_arr, In_arr, It_arr, Ic_arr], axis=1, keys=['G', 'A', 'B', 'In', 'It', 'Ic'])
X = X.fillna(X.mean())

y = f_arr

model = RandomForestRegressor()
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[-5:]

plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(indices)), importances[indices], color='skyblue', edgecolor='darkblue')
plt.yticks(np.arange(len(indices)), X.columns[indices], fontsize=12)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('5 Best Feature Selection Scores', fontsize=16)
plt.grid(axis='x')

plt.show()

'''
Draw the correlation heatmap for 5 most related variables (G, A, In, It, Ic) and the mortality (m)
'''

# Ensure all arrays are 1-dimensional
G_arr = np.array(G_arr, dtype=float).flatten()
A_arr = np.array(A_arr, dtype=float).flatten()
In_arr = np.array(In_arr, dtype=float).flatten()
It_arr = np.array(It_arr, dtype=float).flatten()
Ic_arr = np.array(Ic_arr, dtype=float).flatten()
m_arr = np.array(m_arr, dtype=float).flatten()

# Find the minimum length of the arrays
min_length = min(len(G_arr), len(A_arr), len(In_arr), len(It_arr), len(Ic_arr), len(m_arr))

# Truncate all arrays to the minimum length
G_arr = G_arr[:min_length]
A_arr = A_arr[:min_length]
In_arr = In_arr[:min_length]
It_arr = It_arr[:min_length]
Ic_arr = Ic_arr[:min_length]
m_arr = m_arr[:min_length]

# Combine all variables into a DataFrame
df = pd.DataFrame({
    'G': G_arr,
    'A': A_arr,
    'In': In_arr,
    'It': It_arr,
    'Ic': Ic_arr,
    'm': m_arr
})

# Calculate the Spearman correlation matrix
corr = df.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Correlation coefficient'})
plt.title('Spearman correlation Heatmap')
plt.show()

'''
Calculate the average mortality in gender
'''

min_length = min(len(G_arr), len(m_arr_scaled))
G_arr = G_arr[:min_length]
m_arr_scaled = m_arr_scaled[:min_length]

#  Create a DataFrame with gender and mortality
df = pd.DataFrame({
    'Gender': G_arr,
    'Mortality': m_arr_scaled
})

df['GenderGroup'] = df['Gender'].map({2.6: 'Male', 2: 'Female'})

# Calculate the average mortality
grouped_gender = df.groupby('GenderGroup')['Mortality'].mean()

# Gender Group plot
fig, ax = plt.subplots(figsize=(10, 6))
grouped_gender.plot(kind='bar', color=['skyblue', 'lightgreen'])
ax.set_xlabel('Gender Group', fontsize=12)
ax.set_ylabel('Average Mortality', fontsize=12)
ax.set_title('Average Mortality by Gender Group', fontsize=14)
ax.grid(True)
ax.legend()
plt.show()