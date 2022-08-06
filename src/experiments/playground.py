import torch
from icecream import ic
import os.path as osp
from src.experiments.esnli import get_loader
from src.models.kax import KAX
from tqdm import tqdm

from src.settings import settings
from src.utils.embeddings import tokenizer

# Get test dataloader
dataloader = get_loader('test')

# Load model
model = KAX().to(settings.device)
results_path = ''  # Insert path to model
model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
model.eval()

explanations = []
# Run model on test set
for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
    att_knwl, _, _ = model(inputs)
    encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
    encoded_knowledge = {'knowledge_embedding': att_knwl.knowledge}
    nles_tokens = model.explainer.model.generate(**encoded_inputs, **encoded_knowledge, do_sample=False, max_length=30)
    explanations.append(tokenizer.batch_decode(nles_tokens, skip_special_tokens=True))

# Save explanations to text file
with open(osp.join(results_path, 'explanations.txt'), 'w') as f:
    for explanation in explanations:
        f.write(explanation + '\n')

ic(explanations)
