"""Strict one-to-one truth/finding matching; all amounts are minor units."""
from dataclasses import dataclass,asdict

@dataclass
class Score:
    tp:int
    fp:int
    fn:int
    precision:float
    recall:float
    f1:float
    reward:float
    agent_reward:float
    valid:bool=True
    error:str=''
    def to_dict(self):return asdict(self)

def _matches(truth,found):
    try:
        ids=found['affected_ids'];evidence=found['evidence_ids']
        if not isinstance(ids,list) or len(ids)!=len(set(map(str,ids))):return False
        return (truth['type']==found['type'] and sorted(map(str,truth['affected_ids']))==sorted(map(str,ids))
            and isinstance(found['claimed_amount_minor'],int) and not isinstance(found['claimed_amount_minor'],bool)
            and abs(truth['expected_amount_minor']-found['claimed_amount_minor'])<=truth.get('amount_tolerance_minor',0)
            and truth['currency']==found['currency'] and isinstance(evidence,list)
            and any(set(map(str,group)).issubset(set(map(str,evidence))) for group in truth['acceptable_evidence_sets']))
    except (KeyError,TypeError,ValueError):return False

def calculate_from_submission(truth,submission,inspection_steps=0,mode='open'):
    issues=truth.get('issues',[])
    if (not isinstance(submission,dict) or submission.get('episode_id')!=truth.get('episode_id')
        or not isinstance(submission.get('findings'),list) or len(submission['findings'])>30
        or any(not isinstance(x,dict) for x in submission['findings'])):
        return Score(0,0,len(issues),0,0,0,-1.0,0.0 if mode=='closed' else -1.0,False,'invalid submission')
    findings=submission['findings']; matched=set();tp=0
    for f in findings:
        for idx,t in enumerate(issues):
            if idx not in matched and _matches(t,f):
                matched.add(idx);tp+=1;break
    fp=len(findings)-tp;fn=len(issues)-tp
    p=tp/(tp+fp) if tp+fp else (1.0 if not issues else 0.0)
    r=tp/(tp+fn) if tp+fn else 1.0
    f1=2*p*r/(p+r) if p+r else 0.0
    reward=max(-1.0,min(1.0,2*f1-1-0.002*max(0,inspection_steps)))
    return Score(tp,fp,fn,p,r,f1,reward,0.0 if mode=='closed' else reward)
