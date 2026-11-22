import csv
import json
import random
import argparse
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

def generate_company(seed=42, num_invoices=24):
    rng = random.Random(seed)
    names = ['Astana Tech','Nomad Finance','Almaty Solutions','Steppe Digital','KazTech Group']
    company = {'company_name':rng.choice(names),'base_currency':'USD','seed':seed,'invoices':[],'payments':[],'ledger':[],'fx':[]}
    entry_no=1; payment_no=1
    def add(tx, day, inv, cur, entries):
        nonlocal entry_no
        for acc, debit, credit in entries:
            company['ledger'].append({'entry_id':f'LED-{entry_no:05d}','transaction_id':tx,'date':day,'invoice_id':inv,'account':acc,'debit':debit,'credit':credit,'currency':cur})
            entry_no+=1
    for i in range(1,num_invoices+1):
        inv=f'INV-{i:04d}'; cur=rng.choice(['USD','EUR','KZT'])
        amount=rng.randint(100,1000)*1000 if cur=='KZT' else rng.randint(500,5000)
        day=date(2026,1,1)+timedelta(days=rng.randint(0,300))
        company['invoices'].append({'invoice_id':inv,'customer_id':f'CUST-{rng.randint(1,8):03d}','amount':amount,'currency':cur,'invoice_date':day.isoformat(),'due_date':(day+timedelta(days=30)).isoformat(),'description':'Business services'})
        add(f'ISSUE-{inv}',day.isoformat(),inv,cur,[('Accounts Receivable',amount,0),('Service Revenue',0,amount)])
        amounts=[] if i%6==0 else ([amount//2,amount-amount//2] if i%4==0 else [amount])
        for amt in amounts:
            pid=f'PAY-{payment_no:04d}'; pday=(day+timedelta(days=rng.randint(1,30))).isoformat()
            company['payments'].append({'payment_id':pid,'invoice_id':inv,'amount':amt,'currency':cur,'payment_date':pday})
            add(f'RECEIPT-{pid}',pday,inv,cur,[('Cash',amt,0),('Accounts Receivable',0,amt)])
            payment_no+=1
    for month in range(1,13):
        day=date(2026,month,1).isoformat()
        for cur,rate in {'USD':1.0,'EUR':round(rng.uniform(1.05,1.20),6),'KZT':round(rng.uniform(.0018,.0023),6)}.items():
            company['fx'].append({'date':day,'from_currency':cur,'to_currency':'USD','rate':rate})
    return company

def validate_company(company):
    totals=defaultdict(lambda:[0,0])
    for e in company['ledger']:
        t=totals[e['transaction_id'],e['currency']];t[0]+=e['debit'];t[1]+=e['credit']
    assert all(a==b for a,b in totals.values()),'Unbalanced ledger'
    invs={x['invoice_id']:x for x in company['invoices']}; paid=defaultdict(int)
    for p in company['payments']:
        assert p['invoice_id'] in invs
        assert p['currency']==invs[p['invoice_id']]['currency']
        assert p['payment_date']>=invs[p['invoice_id']]['invoice_date']
        paid[p['invoice_id']]+=p['amount']
    assert all(paid[k]<=v['amount'] for k,v in invs.items())
    return True

def write_jsonl(path,rows):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def write_csv(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save_company(company,output_dir):
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=True)
    write_jsonl(p/'invoices.jsonl',company['invoices']);write_jsonl(p/'payments.jsonl',company['payments'])
    write_csv(p/'ledger.csv',company['ledger'],['entry_id','transaction_id','date','invoice_id','account','debit','credit','currency'])
    write_csv(p/'fx.csv',company['fx'],['date','from_currency','to_currency','rate'])
    policy={'base_currency':'USD','reporting_period':{'start':'2026-01-01','end':'2026-12-31'},'invoice_payment_terms_days':30,'fx_rate_convention':'Monthly reference rates to USD; not yet applied to ledger conversion'}
    manifest={'project':'AuditForge','company_name':company['company_name'],'seed':company['seed'],'episode_id':f"episode_{company['seed']}",'files':['ledger.csv','invoices.jsonl','payments.jsonl','fx.csv','policy.json'],'invoice_count':len(company['invoices']),'payment_count':len(company['payments']),'ledger_entry_count':len(company['ledger'])}
    for name,content in [('policy.json',policy),('manifest.json',manifest),('company.json',company)]:
        (p/name).write_text(json.dumps(content,indent=2),encoding='utf-8')
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output');a=ap.parse_args()
    company=generate_company(a.seed);validate_company(company);assert company==generate_company(a.seed)
    out=save_company(company,a.output or f'data/episode_{a.seed}')
    print(f'AuditForge generator completed! Invoices: {len(company["invoices"])} Payments: {len(company["payments"])} Ledger entries: {len(company["ledger"])}')
    print('Accounting validation: PASSED | Reproducibility: PASSED | Files:',out)
