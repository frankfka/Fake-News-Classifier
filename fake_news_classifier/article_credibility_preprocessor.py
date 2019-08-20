import pandas as pd

# Get original df
original_data = pd.read_pickle('./data/json_data.pkl')

# Keys
ID = 'id'
CLAIM = 'claim'
CLAIMANT = 'claimant'
LABEL = 'label'
REL_ARTICLES = 'related_articles'
DATE = 'date'

# Stores the final data
data_dict = {
    ID: [],
    CLAIM: [],
    CLAIMANT: [],
    LABEL: [],
    REL_ARTICLES: [],
    DATE: []
}

# Loop through original data and add to new data dict
for idx, row in original_data.iterrows():
    # Get all info
    claim_id = row[ID]
    claim = row[CLAIM]
    claimant = row[CLAIMANT]
    label = row[LABEL]
    date = row[DATE]
    related_articles = row[REL_ARTICLES]
    for article_id in related_articles:
        # Append each article ID to a separate data entry - use a list to be compatible with all other code
        data_dict[ID].append(claim_id)
        data_dict[CLAIM].append(claim)
        data_dict[CLAIMANT].append(claimant)
        data_dict[LABEL].append(label)
        data_dict[DATE].append(date)
        data_dict[REL_ARTICLES].append([article_id])

# Create a new dataframe
new_data = pd.DataFrame(data=data_dict)
print(new_data.head())
new_data.to_pickle('./data/json_data_individual.pkl')