import evaluate
from bert_score import score as bert_score
from bleurt import score as bleurt_score

candidates = ['hey']
refs = ['hi']
bert_score(candidates, refs, lang='en', return_hash=True)
meteor = evaluate.load('meteor', module_type='metric')
meteor.compute(predictions=candidates, references=refs)
bleurt_scorer = bleurt_score.BleurtScorer()
bleurt_scorer.score(candidates=candidates, references=refs)
