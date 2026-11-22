import tempfile,unittest
from generator.data_generator import generate_company,validate_company
from generator.error_generator import build_episode,write_episode
from environment.auditor_env import AuditorEnv
from evaluator.reward import calculate_from_submission

class Integration(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(generate_company(42),generate_company(42))
        self.assertEqual(build_episode(42),build_episode(42))
        self.assertNotEqual(build_episode(42),build_episode(43))
    def test_balance(self):self.assertTrue(validate_company(generate_company(42)))
    def test_four_errors(self):
        b=build_episode(42);self.assertEqual({x['type'] for x in b['truth']['issues']},{'duplicate_invoice','payment_allocation','fx_error','period_cutoff'})
    def test_env_files(self):
        with tempfile.TemporaryDirectory() as d:
            b=build_episode(42);write_episode(b,d)
            with AuditorEnv(d,training_mode=False) as env:
                obs,_=env.reset(seed=42)
                self.assertNotIn('truth',str(obs))
                for table in ('invoices','payments','ledger','fx'):
                    out,*_=env.step({'type':'list_records','table':table})
                    self.assertGreater(out['total'],0)
                payload={'episode_id':obs['episode_id'],'findings':[]}
                out,rew,term,trunc,_=env.step({'type':'submit','findings':payload})
                self.assertTrue(term);self.assertEqual(out.get('message'),'submission_received')
    def test_reward(self):
        t=build_episode(42)['truth'];good=[]
        for x in t['issues']:
            good.append({'type':x['type'],'affected_ids':x['affected_ids'],'claimed_amount_minor':x['expected_amount_minor'],'currency':x['currency'],'evidence_ids':x['acceptable_evidence_sets'][0],'explanation':'test'})
        perfect=calculate_from_submission(t,{'episode_id':t['episode_id'],'findings':good})
        self.assertEqual(perfect.tp,4);self.assertEqual(perfect.f1,1)
        dup=calculate_from_submission(t,{'episode_id':t['episode_id'],'findings':good+[good[0]]})
        self.assertEqual(dup.tp,4);self.assertEqual(dup.fp,1)
        clean={'episode_id':'clean','issues':[]}
        self.assertEqual(calculate_from_submission(clean,{'episode_id':'clean','findings':[]}).f1,1)
        self.assertEqual(calculate_from_submission(t,{'episode_id':t['episode_id'],'findings':'oops'}).reward,-1)
class AuditAgent(unittest.TestCase):
    def test_basic_finds_all_four_across_seeds(self):
        from agents.baselines import basic_auditor
        for seed in (1,2,7,42,100,2026):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as d:
                b=build_episode(seed);write_episode(b,d)
                with AuditorEnv(d,training_mode=False) as env:
                    obs,_=env.reset(seed=seed)
                    findings=basic_auditor(obs,env)
                    submission={'episode_id':obs['episode_id'],'findings':findings}
                    out,_,terminated,_,_=env.step({'type':'submit','findings':submission})
                    self.assertTrue(terminated)
                    self.assertEqual(out.get('message'),'submission_received',out)
                    score=calculate_from_submission(b['truth'],submission,env.steps-1)
                    self.assertEqual(score.f1,1.0,score.to_dict())

if __name__=='__main__':unittest.main()
