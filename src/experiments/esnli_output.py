import os.path as osp

import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.esnli import get_loader
from src.models.kax import KAX
from src.settings import settings
from src.utils.embeddings import tokenizer

# Get test dataloader
dataloader = get_loader('test')

# Load model
model = KAX().to(settings.device)
results_path = settings.input_dir
model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
model.eval()

sentences = []
gold_labels = []
predictions = []
explanations = []

# Run model on test set
for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
    # Run model
    knwl, fused_knwl, _, pred = model(inputs)

    # Get predictions and explanations
    encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
    encoded_knowledge = {'knowledge_embedding': fused_knwl.fused}
    nles_tokens = model.explainer.model.generate(**encoded_inputs, **encoded_knowledge, do_sample=False, max_length=30)
    sentences.extend(tokenizer.batch_decode(encoded_inputs['input_ids'], skip_special_tokens=True))
    explanations.extend(tokenizer.batch_decode(nles_tokens, skip_special_tokens=True))
    gold_labels.extend(inputs['gold_label'].tolist())
    predictions.extend(pred.logits.argmax(1).tolist())

# Save results
results = {'sentence': sentences, 'gold_label': gold_labels, 'prediction': predictions, 'explanation': explanations}
results = pd.DataFrame(results)
results.to_csv(osp.join(results_path, 'test_results.csv'), index=False)
