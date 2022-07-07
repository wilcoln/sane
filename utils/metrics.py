from bert_score import score

cands = ['My name is Wilfried', 'The boat is down']

refs = ['Wilfried is my name', 'The lion is sleeping']

(P, R, F), hashname = score(cands, refs, lang="en", return_hash=True)
print(f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
