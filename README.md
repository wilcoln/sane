# ox-msc-project

Oxford MSc in Advanced Computer Science Project

## Todos:

1. Debug The model
    - Right now NoKnowledge is better, we need to beat it
    - We need to see performance improvement over noisy knowledge

2. Evaluation of faithfulness of explanations (1h)
    1. Feature importance agreement
    2. Robustness equivalence
3. Human evaluation of explanations

###

4. Use different knowledge representations

- Get the generated explanations by rexc on the test set from Bodhi - asked
- Compare outputs on an overlapping sample of size 50
- Read the e-vil paper for human evaluation
- Tell why the model is better e.g. hallucinate less,
- Implement various knowledge representation methods
- Grid search for hyperparameters.
    - Knowledge Dimension
    - learning rate
    - dropout
    - batch size
    - knowledge representation

- alpha for combining explanation loss with prediction loss
- Selective attention on batched knowledge
- Read EMNLP paper
- Complexity discussion like in the "Attention is All you need" paper

# Ideas

- Random sampling of the kg around anchor nodes instead of just random sampling the kg

## Done:

- Choose best model using validation set.
- Save model at each epoch
- Give better name to results directory.
- Implement conceptnet api
- add parameter for result dir naming
- what embedding to use ? avg bart, sent_trans, bert cls token

5. Attention maps reporting (as a heatmap)st of required experiments: (1h)
6. Automatic evaluation of explanations (1h)

- Tried using cls token, very poor !, token mean performs ok, almost on par with sentence transformer
- Combine explanations and input for prediction ?