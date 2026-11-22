"""Deterministic auditing agents using ONLY public environment actions."""
from collections import defaultdict
from decimal import Decimal


def fetch(env, table):
    rows=[]; cursor=None
    while env.steps < env.max_steps-2:
        obs,_,term,trunc,_=env.step({'type':'list_records','table':table,'cursor':cursor})
        if term or trunc or 'error' in obs: break
        rows.extend(obs['records']);cursor=obs['next_cursor']
        if cursor is None:break
    return rows


def silent(obs,env):return []

def noisy(obs,env):
    rows=fetch(env,'invoices')
    return [{'type':'duplicate_invoice','affected_ids':[x['invoice_id']],'claimed_amount_minor':0,'currency':x['currency'],'evidence_ids':[x['invoice_id']],'explanation':'Unverified suspicion'} for x in rows[:12]]

def duplicate_only(obs,env):
    rows=fetch(env,'invoices');groups={};findings=[]
    for inv in rows:
        key=(inv.get('customer_id'),inv.get('amount'),inv.get('currency'),inv.get('invoice_date'),inv.get('description'))
        if key in groups:
            prev=groups[key]
            findings.append({'type':'duplicate_invoice','affected_ids':[prev['invoice_id'],inv['invoice_id']],'claimed_amount_minor':int(inv['amount'])*100,'currency':inv['currency'],'evidence_ids':[prev['invoice_id'],inv['invoice_id']],'explanation':'Duplicate invoice content with distinct IDs'})
        else:groups[key]=inv
    return findings

def basic_auditor(obs,env):
    invoices=fetch(env,'invoices');payments=fetch(env,'payments');ledger=fetch(env,'ledger');fx=fetch(env,'fx')
    findings=[];seen=set();invoice_by_id={x['invoice_id']:x for x in invoices}
    def add(kind,affected,amount,currency,evidence,explanation):
        key=(kind,tuple(sorted(affected)))
        if key not in seen:
            seen.add(key);findings.append({'type':kind,'affected_ids':affected,'claimed_amount_minor':int(amount),'currency':currency,'evidence_ids':evidence,'explanation':explanation})
    groups={}
    for inv in invoices:
        key=(inv.get('customer_id'),inv.get('amount'),inv.get('currency'),inv.get('invoice_date'),inv.get('description'))
        if key in groups:
            first=groups[key]
            add('duplicate_invoice',[first['invoice_id'],inv['invoice_id']],int(inv['amount'])*100,inv['currency'],[first['invoice_id'],inv['invoice_id']],'Matching customer, date, amount and description')
        else:groups[key]=inv
    by_transaction=defaultdict(list)
    for row in ledger:by_transaction[row['transaction_id']].append(row)
    for payment in payments:
        entries=by_transaction.get('RECEIPT-'+payment['payment_id'],[])
        if entries and any(e['invoice_id']!=payment['invoice_id'] for e in entries):
            original=entries[0]['invoice_id']
            add('payment_allocation',[original,payment['payment_id']],int(payment['amount'])*100,payment['currency'],[payment['payment_id'],original],'Payment document allocation differs from receipt journal')
    approved={(r['date'],r['from_currency'],r['to_currency']):Decimal(str(r['rate'])) for r in obs.get('policy',{}).get('approved_fx_rates',[])}
    for row in fx:
        key=(row.get('date'),row.get('from_currency'),row.get('to_currency'))
        if key in approved and abs(Decimal(str(row['rate']))-approved[key])>Decimal('0.000001'):
            eid=row.get('evidence_id') or '|'.join(key)
            add('fx_error',[eid],500,'USD',[eid],'Recorded FX rate differs from independently approved reference rate')
    period=obs.get('policy',{}).get('reporting_period',{})
    start,end=period.get('start',''),period.get('end','')
    for row in ledger:
        inv=invoice_by_id.get(row.get('invoice_id'))
        if row.get('transaction_id','').startswith('ISSUE-') and inv and row.get('account')=='Accounts Receivable':
            if row['date']!=inv['invoice_date'] and (row['date']<start or row['date']>end):
                add('period_cutoff',[row['entry_id']],int(row['debit'])*100,row['currency'],[row['entry_id'],row['invoice_id']],'Posting date falls outside reporting period while invoice is in-period')
    return findings

BASELINES={'silent':silent,'noisy':noisy,'duplicate_only':duplicate_only,'basic':basic_auditor}
