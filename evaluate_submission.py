import argparse,json
from pathlib import Path
from evaluator.reward import calculate_from_submission

def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--submission',required=True);p.add_argument('--inspection-steps',type=int,default=0);p.add_argument('--mode',choices=['open','closed'],default='open');a=p.parse_args()
    truth=json.loads(Path(a.truth).read_text());submission=json.loads(Path(a.submission).read_text())
    print(json.dumps(calculate_from_submission(truth,submission,a.inspection_steps,a.mode).to_dict(),indent=2))
if __name__=='__main__':main()
