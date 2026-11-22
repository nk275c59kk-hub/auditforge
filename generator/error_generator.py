"""Deterministic synthetic corruption; hidden truth stays outside public/ directory."""
import copy
import json
import random
from pathlib import Path
from generator.data_generator import generate_company, validate_company, write_csv, write_jsonl

def build_episode(seed=42, config=None):
    config=config or {}; rng=random.Random(seed+1000003)
    base=generate_company(seed,num_invoices=int(config.get('num_invoices',24)))
    validate_company(base)
    public=copy.deepcopy(base)
    inv=public['invoices']; pay=public['payments']; ledger=public['ledger']; fx=public['fx']
    issues=[]; provenance=[]; ep_id=f'EP-{seed:04d}'
    def issue(kind, affected, amount_minor, currency, evidence, before, after):
        issue_id=f'ISS-{len(issues)+1:03d}'
        issues.append({'issue_id':issue_id,'type':kind,'affected_ids':affected,'expected_amount_minor':int(amount_minor),'currency':currency,'amount_tolerance_minor':0,'acceptable_evidence_sets':[evidence]})
        provenance.append({'issue_id':issue_id,'type':kind,'before':before,'after':after})
    # 1. duplicate document, same customer, amount and invoice date, new document ID
    original=inv[1]; duplicate=copy.deepcopy(original);duplicate['invoice_id']='INV-DUP-0001'
    inv.append(duplicate)
    issue('duplicate_invoice',[original['invoice_id'],duplicate['invoice_id']],original['amount']*100,original['currency'],[original['invoice_id'],duplicate['invoice_id']],None,duplicate)
    # 2. a payment allocated to a different invoice, while ledger keeps the original allocation
    payment=pay[0]; before=copy.deepcopy(payment)
    original_inv=payment['invoice_id']; other=next(x for x in inv if x['invoice_id']!=original_inv and x['currency']==payment['currency'])
    payment['invoice_id']=other['invoice_id']
    issue('payment_allocation',[original_inv,payment['payment_id']],payment['amount']*100,payment['currency'],[payment['payment_id'],original_inv],before,copy.deepcopy(payment))
    # 3. corrupt FX reference rate; independent approved rates in policy allow verification
    rate=next(x for x in fx if x['from_currency']=='EUR'); before=copy.deepcopy(rate)
    approved_fx=[dict(x) for x in fx]
    rate['rate']=round(rate['rate']+0.05,6)
    rate['evidence_id']=f"{rate['date']}|{rate['from_currency']}|{rate['to_currency']}"
    issue('fx_error',[rate['evidence_id']],500,'USD',[rate['evidence_id']],before,copy.deepcopy(rate))
    # 4. posting outside the reporting period, without changing document date
    posting=next(x for x in ledger if x['transaction_id'].startswith('ISSUE-'))
    before=copy.deepcopy(posting);posting['date']='2027-01-01'
    issue('period_cutoff',[posting['entry_id']],posting['debit']*100,posting['currency'],[posting['entry_id'],posting['invoice_id']],before,copy.deepcopy(posting))
    # FX rows must all have evidence IDs for consistent submissions
    for row in fx:
        row.setdefault('evidence_id',f"{row['date']}|{row['from_currency']}|{row['to_currency']}")
    episode={'manifest':{'project':'AuditForge','episode_id':ep_id,'company_name':base['company_name'],'base_currency':'USD','schema_version':'auditforge/1','n_invoices':len(inv),'n_payments':len(pay),'n_ledger_rows':len(ledger)},'policy':{'base_currency':'USD','reporting_period':{'start':'2026-01-01','end':'2026-12-31'},'payment_terms_days':30,'fx_rate_convention':'Monthly USD reference rate', 'approved_fx_rates':approved_fx},'ledger':ledger,'invoices':inv,'payments':pay,'fx':fx}
    return {'episode':episode,'truth':{'episode_id':ep_id,'issues':issues},'provenance':{'episode_id':ep_id,'changes':provenance}}

def write_episode(result,output_dir):
    root=Path(output_dir);pub=root/'public';priv=root/'private';pub.mkdir(parents=True,exist_ok=True);priv.mkdir(parents=True,exist_ok=True)
    ep=result['episode']
    write_csv(pub/'ledger.csv',ep['ledger'],['entry_id','transaction_id','date','invoice_id','account','debit','credit','currency'])
    write_csv(pub/'fx.csv',ep['fx'],['date','from_currency','to_currency','rate','evidence_id'])
    write_jsonl(pub/'invoices.jsonl',ep['invoices']);write_jsonl(pub/'payments.jsonl',ep['payments'])
    for name,obj in [('manifest.json',ep['manifest']),('policy.json',ep['policy'])]:
        (pub/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    for name,obj in [('truth.json',result['truth']),('provenance.json',result['provenance'])]:
        (priv/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return root

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output');args=ap.parse_args()
    root=write_episode(build_episode(args.seed),args.output or f'runs/episode_{args.seed}')
    print('Generated:',root)
