# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading data by reading in the titanic_train.csv and titanic_test.csv files into pandas dataframes.

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

train.shape

test.head()

test.shape

# Checking for duplicates
print("In the train dataset, there are {} duplicates.".format(train.duplicated().sum())) 
print("In the test dataset, there are {} duplicates.".format(test.duplicated().sum()))

## Exploratory Data Analysis
# Creating a heatmap to visualize missing values in the train dataset
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='Greens')
# Adding title
plt.title('Missing Values in Train Dataset')
plt.show()

# Creating a heatmap to visualize missing values in the test dataset
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='Greens')
# Adding title
plt.title('Missing Values in Test Dataset')
plt.show()

"""Around 20 percent of the Age data is absent. This proportion of missing Age values is likely manageable for replacement through imputation techniques. 
However, upon examining the Cabin column, it appears that a substantial portion of the data is missing, making it less useful for basic analysis. 
We may opt to drop this column later or transform it into another feature, such as a binary indicator denoting whether the cabin information is known (1) or not (0).
Let's continue visualizing the data. Moving forward, I will focus on analyzing only the training dataset while making corresponding adjustments in both datasets."""

sns.set(style="whitegrid", palette="muted", color_codes=True)
# Create a count plot to visualize the distribution of the 'Survived' column in the train DataFrame
sns.countplot(x='Survived',data=train)
# Add title and labels 
plt.title('Distribution of Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Plotting survival counts based on gender
sns.countplot(x='Survived', hue='Sex', data=train)
# Adding title and labels
plt.title('Survival Counts by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Plotting survival counts based on passenger class
sns.countplot(x='Survived', hue='Pclass', data=train)
# Adding title and labels
plt.title('Survival Counts by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Creating a histogram to visualize the distribution of ages
sns.histplot(train['Age'].dropna(), kde=False, bins=30)
# Adding title and labels
plt.title('Distribution of Passengers Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Creating a count plot to visualize the number of siblings/spouses aboard
sns.countplot(x='SibSp', data=train, color='g')
# Adding title and labels
plt.title('Number of Siblings/Spouses Aboard')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Count')
plt.show()

# Creating a histogram to visualize the distribution of fares
train['Fare'].hist(color='r', bins=40, figsize=(8,4))
# Adding title and labels
plt.title('Distribution of Fare Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Looking for correlations
corr_matrix = train.corr(numeric_only=True)
# Increase the figure size
plt.figure(figsize=(12, 10))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Choose a diverging color scheme
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, annot=True, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

"""Key Insights from the Visualizations:
1. The dataset shows a higher count of survivors compared to non-survivors.
2. However, the survival rate is significantly higher among women compared to men.
3. Among the deceased, there's a disproportionate number from the third class.
4. The age distribution of passengers is centered around 20-35 years old.
5. A majority of passengers traveled without any siblings or spouses, with around 25% traveling with one sibling or spouse, indicating primarily single passengers and some couples.
6. The distribution of fare prices is right-skewed, indicating a few very expensive tickets and the majority priced under 40.
7. There's an average negative correlation between passenger class and fare, as expected, with first class being the most expensive and third class the least.
8. Additionally, there's a positive correlation between the number of siblings/spouses and the number of parents/children, suggesting that married couples are likely to travel with children."""