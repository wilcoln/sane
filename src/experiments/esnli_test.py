import os.path as osp
import time

import pandas as pd
import torch
from tqdm import tqdm

from src.datasets.esnli import get_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.embeddings import tokenizer
from src.utils.format import fmt_stats_dict


@torch.no_grad()
def test(model, results_path, dataloader):
    sentences = []
    gold_labels = []
    predictions = []
    explanations = []

    # Set test loss value
    test_loss = 0.0

    # Reset values for accuracy computation
    correct = 0
    total = 0
    test_time = 0
    # Run model on test set
    for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        # start batch time
        batch_start = time.time()
        # Run model
        out = model(inputs)
        pred, nle = out[:2]
        att_knwl = None if settings.no_knowledge else out[2]

        # Compute loss
        loss = settings.alpha * nle.loss + (1 - settings.alpha) * pred.loss

        # Update Loss
        test_loss += loss.item()

        # Update Accuracy
        predicted = pred.logits.argmax(1)
        total += inputs['gold_label'].size(0)
        correct += predicted.eq(inputs['gold_label'].to(settings.device)).sum().item()
        test_time += time.time() - batch_start

        # Get predictions and explanations
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_knowledge = {} if settings.no_knowledge else {'knowledge_embedding': att_knwl.output} 
        nles_tokens = model.explainer.model.generate(**encoded_inputs, **encoded_knowledge, do_sample=False,
                                                     max_length=30)
        sentences.extend(tokenizer.batch_decode(encoded_inputs['input_ids'], skip_special_tokens=True))
        explanations.extend(tokenizer.batch_decode(nles_tokens, skip_special_tokens=True))
        gold_labels.extend(inputs['gold_label'].tolist())
        predictions.extend(pred.logits.argmax(1).tolist())

    test_loss /= len(dataloader)
    test_acc = 100. * correct / total
    stats_dict = {'test_acc': test_acc, f'test_loss': test_loss, f'test_time': test_time}

    # Print stats
    print(fmt_stats_dict(stats_dict))

    # Save results
    results = {'sentence': sentences, 'gold_label': gold_labels, 'prediction': predictions, 'explanation': explanations}
    results = pd.DataFrame(results)
    results.to_csv(osp.join(results_path, f'test_results{settings.out_suffix}.csv'), index=False)


if __name__ == '__main__':
    # Get test dataloader
    dataloader = get_loader('test')

    # Load model
    model = SANENoKnowledge() if settings.no_knowledge else SANE()
    model = model.to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()

    # test model
    test(model, results_path, dataloader)
