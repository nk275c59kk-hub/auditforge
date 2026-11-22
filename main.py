import argparse,json,unittest
from pathlib import Path
from generator.error_generator import build_episode,write_episode
from environment.auditor_env import AuditorEnv
from evaluator.reward import calculate_from_submission
from agents.baselines import BASELINES

def demo(seed=42,agent='basic'):
    result=build_episode(seed);path=write_episode(result,f'runs/episode_{seed}')
    with AuditorEnv(path,training_mode=False) as env:
        obs,info=env.reset(seed=seed)
        findings=BASELINES[agent](obs,env)
        payload={'episode_id':obs['episode_id'],'findings':findings}
        remaining=max(0,env.max_steps-env.steps)
        observation,reward,terminated,truncated,info=env.step({'type':'submit','findings':payload}) if remaining else ({'error':'step limit'},0,False,True,{})
        score=calculate_from_submission(result['truth'],payload,env.steps-1,mode='closed')
        print('Episode:',obs['episode_id'],'| Agent:',agent,'| Submitted:',observation.get('message',observation.get('error')))
        print('Score:',json.dumps(score.to_dict(),indent=2))
        print('Public data:',path/'public','| Private truth (never pass to agent):',path/'private')
        return score

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['demo','generate','test']);p.add_argument('--seed',type=int,default=42);p.add_argument('--agent',choices=BASELINES,default='basic');a=p.parse_args()
    if a.command=='demo':demo(a.seed,a.agent)
    elif a.command=='generate':print(write_episode(build_episode(a.seed),f'runs/episode_{a.seed}'))
    else:
        result=unittest.defaultTestLoader.discover('tests');out=unittest.TextTestRunner(verbosity=2).run(result)
        if not out.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
