"""Build deterministic train/dev episodes and public-only agent trajectories."""
import argparse,csv,json
from pathlib import Path
from generator.error_generator import build_episode,write_episode
from environment.auditor_env import AuditorEnv
from evaluator.reward import calculate_from_submission
from agents.baselines import BASELINES

def build(output='submission',train=10,dev=10):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split,seeds in [('train',range(1000,1000+train)),('dev',range(2000,2000+dev))]:
        for seed in seeds:
            episode=build_episode(seed)
            path=write_episode(episode,root/split/f'episode_{seed}')
            for name,agent in BASELINES.items():
                trace=root/'trajectories'/split/f'episode_{seed}_{name}.jsonl'
                with AuditorEnv(path,training_mode=False,trajectory_path=trace) as env:
                    obs,_=env.reset(seed=seed)
                    findings=agent(obs,env)
                    submission={'episode_id':obs['episode_id'],'findings':findings}
                    result,_,terminated,truncated,_=env.step({'type':'submit','findings':submission})
                    score=calculate_from_submission(episode['truth'],submission,env.steps-1,mode='closed')
                    rows.append({'split':split,'seed':seed,'agent':name,'tp':score.tp,'fp':score.fp,'fn':score.fn,'f1':score.f1,'reward':score.reward,'steps':env.steps,'valid':score.valid and result.get('message')=='submission_received','terminated':terminated,'truncated':truncated})
    with (root/'baseline_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={name:{'episodes':len([r for r in rows if r['agent']==name]),'mean_f1':round(sum(r['f1'] for r in rows if r['agent']==name)/len([r for r in rows if r['agent']==name]),4),'mean_reward':round(sum(r['reward'] for r in rows if r['agent']==name)/len([r for r in rows if r['agent']==name]),4)} for name in BASELINES}
    (root/'baseline_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Built',len(rows),'baseline trajectories in',root)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='submission');p.add_argument('--train',type=int,default=10);p.add_argument('--dev',type=int,default=10);a=p.parse_args();build(a.output,a.train,a.dev)
