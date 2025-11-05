# Exp6: ID3 Decision Tree using Housing Dataset
import pandas as pd
import numpy as np
from math import log2

# 1. Load Dataset
df = pd.read_csv("Housing.csv")

# 2. Convert 'price' into categorical bins
df['price_category'] = pd.cut(df['price'],
                              bins=3,
                              labels=['Low', 'Medium', 'High'])

# Convert continuous features into categories for ID3
df['area_cat'] = pd.cut(df['area'], bins=3, labels=['Small', 'Medium', 'Large'])
df['bedrooms'] = df['bedrooms'].astype(str)
df['bathrooms'] = df['bathrooms'].astype(str)

# Keep relevant columns
df = df[['area_cat', 'bedrooms', 'bathrooms', 'price_category']]

# 3. Entropy Function
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i]/np.sum(counts)) * log2(counts[i]/np.sum(counts))
                   for i in range(len(elements))])

# 4. Information Gain Function
def info_gain(data, split_attribute, target_name="price_category"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i]/np.sum(counts)) *
        entropy(data.where(data[split_attribute] == vals[i]).dropna()[target_name])
        for i in range(len(vals))
    ])
    return total_entropy - weighted_entropy

# 5. ID3 Algorithm
def id3(data, original_data, features, target_attribute_name="price_category", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])
        ]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        ]
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature = features[np.argmax(item_values)]
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# 6. Prediction Function
def predict(query, tree, default='Medium'):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result

# 7. Train the Decision Tree
features = list(df.columns)
features.remove('price_category')
tree = id3(df, df, features, target_attribute_name='price_category')

print("\nGenerated Decision Tree:")
print(tree)

# 8. Test with new sample
test_query = {'area_cat': 'Large', 'bedrooms': '3', 'bathrooms': '2'}
prediction = predict(test_query, tree)
print("\nPredicted Price Category for", test_query, "=>", prediction)
