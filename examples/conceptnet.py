import pandas as pd
import os.path as osp
import json

from utils.settings import settings

concept_path = osp.join(settings.data_dir, 'conceptnet/conceptnet-assertions-5.6.0.csv')
cn = pd.read_csv(concept_path, sep='\t')

# adding column name to the respective columns
cn.columns = ['uri', 'relation', 'source', 'target', 'data']

# Remove uri column
cn.drop(columns=['uri'], inplace=True)

# Keep only rows where language of the source and target is English
cn = cn[(cn['source'].str.contains('/c/en/')) & (cn['target'].str.contains('/c/en/'))]

# Keep only the weight data
cn['weight'] = cn.data.apply(lambda x: json.loads(x)['weight'])

# Drop data column
cn.drop(columns=['data'], inplace=True)

# Drop duplicate rows
cn.drop_duplicates(inplace=True)

cn.to_csv(osp.join(concept_path.replace('.csv', '_cleaned.csv')), index=False)
