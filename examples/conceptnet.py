import pandas as pd
import os.path as osp
import json

from utils.settings import settings


def clean():
    conceptnet_path = osp.join(settings.data_dir, 'conceptnet/conceptnet-assertions-5.6.0.csv')
    cn = pd.read_csv(conceptnet_path, sep='\t')

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

    # Remove prefixes from the relation, source and target
    cn['relation'] = cn['relation'].apply(lambda x: x.replace('/r/', ''))
    cn['source'] = cn['source'].apply(lambda x: x.replace('/c/en/', ''))
    cn['target'] = cn['target'].apply(lambda x: x.replace('/c/en/', ''))

    # # Replace underscores with spaces
    # cn['relation'] = cn['relation'].apply(lambda x: x.replace('_', ' '))
    # cn['source'] = cn['source'].apply(lambda x: x.replace('_', ' '))
    # cn['target'] = cn['target'].apply(lambda x: x.replace('_', ' '))

    # Replace node letter type with word type
    synset_types = [
        ('/n', '/noun'),
        ('/v', '/verb'),
        ('/a', '/adjective'),
        ('/r', '/adverb'),
        ('/s', '/adjective_satellite'),
    ]
    for synset_type, word_type in synset_types:
        cn['source'] = cn['source'].apply(lambda x: x.replace(synset_type, word_type))
        cn['target'] = cn['target'].apply(lambda x: x.replace(synset_type, word_type))

    # Drop duplicate rows
    cn.drop_duplicates(inplace=True)

    cn.to_csv(osp.join(conceptnet_path.replace('.csv', '_cleaned.csv')), index=False)
