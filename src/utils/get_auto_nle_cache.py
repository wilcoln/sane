import evaluate
from bert_score import score as bert_score
from bleurt import score as bleurt_score
from icecream import ic
candidates = ['hey']
refs = ['hi']
ic('load bert score cache')
ic(bert_score(candidates, refs, lang='en', return_hash=True))
ic('load meteor cache')
meteor = evaluate.load('meteor', module_type='metric')
meteor.compute(predictions=candidates, references=refs)
ic('load bleurt score cache')
bleurt_scorer = bleurt_score.BleurtScorer()
bleurt_scorer.score(candidates=candidates, references=refs)
ic('loaded')
