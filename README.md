# ox-msc-project

Oxford MSc in Advanced Computer Science Project

## Todos:

- Implement various knowledge representation methods
  List of required experiments:

1. Report Accuracy on Test Set

- Grid search for hyperparameters.
    - Knowledge Dimension
    - learning rate
    - dropout
    - batch size
    - knowledge representation
- Combine explanations and input for prediction ?
- what embedding to use ? avg bart, sent_trans, bert cls token ?
- alpha for combining explanation loss with prediction loss
- Selective attention on batched knowledge

2. Human evaluation of explanations

3. Automatic evaluation of explanations
4. Evaluation of faithfulness of explanations
5. Attention maps reporting (as a heatmap)
6. Use different knowledge representations

7. Next task:
   Read EMNLP paper

## Done:

- Choose best model using validation set.
- Save model at each epoch
- Give better name to results directory.
- Implement conceptnet api
- add parameter for result dir naming