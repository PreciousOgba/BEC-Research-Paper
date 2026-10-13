"""Primary-seed BEC experiment. Private source data stays on the user's computer.

Deadline deviations: automated curation without a completed human QC audit;
near-duplicate candidates conservatively grouped without relationship review;
one primary training seed. Findings are exploratory and source-confounded.
"""
import argparse,csv,hashlib,json,math,random,re,sqlite3,time,warnings,platform
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import sklearn,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning

MAP=dict(zip('aceopyi','\u0430\u0441\u0435\u043e\u0440\u0443\u0456'))
PLACEHOLDER=re.compile(r'\[[A-Z_]+\]')
AUTO=re.compile(r'return receipt|calendar entry:|log messages:|no ancillary schedules awarded|request create date\s*:|contract \d+ was suspended|you will receive this message each time|this is an automated|system logs confirm|irs system alert',re.I)
LINK_ATTACK=re.compile(r'phishing[ _]link|malicious[ _](?:oauth[ _])?link|secure_auth_link|secure_verification_link|security_measures_link|critical_action_link|account freeze link|secure aws link|secure login|(?:click|navigate|generate|verify|verification).{0,100}(?:link|portal)|(?:qr.{0,30}scan|scan.{0,30}qr|profile installation)',re.I|re.S)
META=re.compile(r'(?:^From:|\bTo:).{0,150}\bOrg:.{0,150}\bAmt:',re.I|re.S)
PERSONAL=re.compile(r'^hi mom\b|^hi.{0,20}\bwe are so bad\b|^hi charlotte,.{0,100}house for dinner|^you said some embarrassing|^ok, so now i feel guilty|^i couldn.t resist|^those are pretty funny|^i knew that was coming before you sent it|^scott, i had not heard from you',re.I|re.S)
PROMO=re.compile(r'llb-mail|to shop these pre-holiday|do you yahoo\?|make a great connection at yahoo|^if one year ago you bought|officeName1234',re.I)

def exclusion(r):
    body=r['body'];text=r['subject']+'\n'+body
    if AUTO.search(text):return 'automated_notice_or_log'
    if META.search(body):return 'generation_metadata_only'
    if r['source']=='synthetic_bec' and LINK_ATTACK.search(body):return 'link_or_QR_payload_scenario'
    if PERSONAL.search(body) or PROMO.search(body):return 'personal_or_promotional_text'
    if re.search(r'\b(?:Dow Jones News Service|Of DOW JONES NEWSWIRES)\b',body):return 'syndicated_news'
    if r['source']=='synthetic_bec' and (body.startswith('Sent from my phone') or not re.search(r'\b(?:please|need|process|transfer|pay|send|confirm|wire|release|reply|provide|remit|initiate|execute|purchase|obtain|procure|requir|mandat|direct|request|fund|bank|credit|update|action|deposit|report|payment)\w*\b',body,re.I)):
        return 'missing_substantive_request_review_hold'
    stripped=re.sub(r'\[[A-Z_]+\]','',body)
    if re.match(r"^['\-\s]*save these dates.*\.doc\s*$",stripped,re.I):return 'attachment_filename_only'
    if body.startswith('Allegro Development Corporation Disclaimer:'):return 'disclaimer_without_substantive_message'
    # Conservative holds: require review where cleaning left mostly signatures/headers.
    if 'Senior Counsel Enron North America Corp.' in body and 'Subject: per my voice message' in body:return 'signature_and_header_residue'
    if 'Subject: FW: This one is good!' in body:return 'forwarded_header_residue'
    if body.startswith('John D. Martin Carr P. Collins Chair'):return 'contact_details_only'
    if body.startswith('FYI. ---') and 'privileged and confidential' in body:return 'disclaimer_only'
    if body.startswith('___') and 'Get your FREE download' in body:return 'footer_only'
    return ''

class UF:
    def __init__(self,n):self.p=list(range(n))
    def root(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def union(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def near_pairs(rows):
    sets=[]
    for r in rows:
        t=' '.join(r['body'].split());sets.append({t[i:i+5] for i in range(max(0,len(t)-4))})
    freq=Counter(g for s in sets for g in s);index=defaultdict(list)
    for i,s in enumerate(sets):
        if not s:continue
        prefix=sorted(s,key=lambda g:(freq[g],g))[:len(s)-math.ceil(.9*len(s))+1]
        candidates={j for g in prefix for j in index[g]}
        for j in sorted(candidates):
            t=sets[j]
            if min(len(s),len(t))/max(len(s),len(t))<.9:continue
            inter=len(s&t);union=len(s)+len(t)-inter
            if 10*inter>=9*union:yield j,i,inter/union
        for g in prefix:index[g].append(i)
        if (i+1)%500==0:print(f'Grouping check: {i+1:,} messages',flush=True)

def metrics(y,p,scores=None):
    y=np.asarray(y);p=np.asarray(p);tp=int(np.sum((y==1)&p));fp=int(np.sum((y==0)&p));fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p))
    result={'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'fpr':fp/(fp+tn) if fp+tn else None,'confusion_matrix':{'TP':tp,'FP':fp,'FN':fn,'TN':tn}}
    if scores is not None:result['average_precision']=float(average_precision_score(y,scores))
    return result

def threshold(y,scores):
    best=None
    for t in np.arange(1,100)/100:
        m=metrics(y,scores>=t);rank=(m['f1'] or 0,-m['fpr'],m['recall'],float(t))
        if best is None or rank>best[0]:best=(rank,float(t),m)
    return best[1],best[2]

def perturb(text,kind,rate,seed,ident):
    rng=random.Random(int(hashlib.sha256(f'{seed}:{ident}:{kind}'.encode()).hexdigest(),16))
    def change(t):
        if kind=='homoglyph':return ''.join(MAP.get(c,c) if c in MAP and rng.random()<rate else c for c in t)
        def word(m):
            w=m.group()
            if rng.random()>=rate:return w
            n=rng.randrange(1,len(w));return w[:n]+'\u200b'+w[n:]
        return re.sub(r'[A-Za-z]{4,}',word,t)
    parts=[];start=0
    for m in PLACEHOLDER.finditer(text):parts.extend([change(text[start:m.start()]),m.group()]);start=m.end()
    parts.append(change(text[start:]));return ''.join(parts)

def normalise(t):
    import unicodedata
    return unicodedata.normalize('NFKC',t).translate(str.maketrans({v:k for k,v in MAP.items()})).translate({ord(c):None for c in '\u200b\u200c\u200d\u2060\ufeff'})

def bootstrap_lr(y,clean,variants,groups):
    unique,inv=np.unique(groups,return_inverse=True);g=len(unique)
    counts=np.zeros((g,4));cols=[(y==1)&clean,(y==0)&clean,(y==1)&~clean,(y==0)&~clean]
    for j,c in enumerate(cols):counts[:,j]=np.bincount(inv,weights=c,minlength=g)
    ptp=np.array([np.bincount(inv,weights=(y==1)&p,minlength=g) for p in variants]).T
    rng=np.random.default_rng(606);values=[];redrawn=0
    while sum(len(v) for v in values)<10000:
        indices=rng.integers(0,g,size=(250,g));c=counts[indices].sum(1);a,b,fn,tn=c.T;pos=a+fn;neg=b+tn;valid=(pos>0)&(neg>0)&((a+b)>0)
        redrawn+=int((~valid).sum());c=c[valid];indices=indices[valid];a,b,fn,tn=c.T;pos=a+fn;neg=b+tn
        pert=ptp[indices].sum(1).mean(1)/pos
        values.append(np.column_stack([a/(a+b),a/pos,2*a/(2*a+b+fn),b/neg,(a/pos-pert)*100]))
    values=np.concatenate(values)[:10000]
    return {'percentile_95_ci':{k:[float(v) for v in np.percentile(values[:,j],[2.5,97.5])] for j,k in enumerate(['precision','recall','f1','fpr','delta_recall_percentage_points'])},'replicates':10000,'redrawn':redrawn,'conditional_on_fixed_split_and_model':True}

def main():
    a=argparse.ArgumentParser();a.add_argument('--pool',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if not args.pool.is_file():a.error('Pool database not found: '+str(args.pool))
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'experiment_pool.jsonl').exists():a.error('Output already exists. Use a different output folder to preserve results.')
    start=time.time();db=sqlite3.connect(args.pool.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute('SELECT id,source,subject,body,token_count,length_bin,group_id FROM candidates WHERE selected=1 ORDER BY id')];db.close()
    held=[];retained=[]
    for r in rows:
        reason=exclusion(r)
        if reason:held.append({'record_id':r['id'],'source':r['source'],'reason':reason})
        else:retained.append(r)
    rows=retained;uf=UF(len(rows));known={};exact={}
    for i,r in enumerate(rows):
        for store,key in [(known,r['group_id']),(exact,r['body'])]:
            if key in store:uf.union(i,store[key])
            else:store[key]=i
    pairs=[]
    for j,i,s in near_pairs(rows):uf.union(i,j);pairs.append((rows[j]['id'],rows[i]['id'],s))
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[uf.root(i)].append(r)
    final=[]
    for members in grouped.values():
        if len({r['source'] for r in members})>1:
            held.extend({'record_id':r['id'],'source':r['source'],'reason':'cross_class_near_duplicate_group'} for r in members);continue
        gid=min(r['id'] for r in members)
        for r in members:r['group_id']=gid;r['label']=int(r['source']=='synthetic_bec');final.append(r)
    rows=sorted(final,key=lambda r:r['id']);y=np.array([r['label'] for r in rows]);groups=np.array([r['group_id'] for r in rows])
    if min(Counter(y).values())<100:raise RuntimeError('Too few messages to form meaningful partitions')
    splitter=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=42)
    folds=list(splitter.split(np.zeros(len(rows)),y,groups));test=folds[0][1];val=folds[1][1];train=np.setdiff1d(np.arange(len(rows)),np.r_[test,val])
    for name,idx in [('train',train),('validation',val),('test',test)]:
        if len(set(y[idx]))!=2:raise RuntimeError('A partition lacks a required class')
        for i in idx:rows[i]['split']=name
    assert not(set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]))
    for r in rows:r['text']=r['subject']+'\n'+r['body']
    with (args.output/'experiment_pool.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (args.output/'split_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','source','group_id','split','label']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    with (args.output/'automated_exclusions.json').open('w',encoding='utf-8') as f:json.dump(held,f)
    with (args.output/'near_duplicate_pairs.json').open('w',encoding='utf-8') as f:json.dump(pairs,f)
    print('Partitions saved. Training eight Logistic Regression configurations...',flush=True)
    texts=[r['text'] for r in rows];v=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,smooth_idf=True,norm='l2')
    xtrain=v.fit_transform([texts[i] for i in train]);xval=v.transform([texts[i] for i in val]);xtest=v.transform([texts[i] for i in test]);best=None;trials=[]
    for c in [.1,1,10,100]:
        for weight in [None,'balanced']:
            model=LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=1000,random_state=42)
            with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);model.fit(xtrain,y[train])
            t,m=threshold(y[val],model.predict_proba(xval)[:,1]);rank=(m['f1'],-m['fpr'],m['recall'],-c,weight is None)
            trials.append({'C':c,'class_weight':weight,'threshold':t,'validation':m});print(f'C={c}, weights={weight}, validation F1={m["f1"]:.4f}',flush=True)
            if best is None or rank>best[0]:best=(rank,model,t,c,weight)
    _,model,t,c,weight=best;joblib.dump({'vectorizer':v,'model':model,'threshold':t},args.output/'lr_model.joblib')
    scores=model.predict_proba(xtest)[:,1];clean=scores>=t;yt=y[test];gt=groups[test];bec=yt==1
    clean_metrics=metrics(yt,clean,scores);conditions={};predictions={'clean':clean.astype(int).tolist()}
    for kind in ['homoglyph','zero_width']:
        for rate in [.1,.05,.2]:
            variants=[];normal_metrics=[];coverage=[];changed=[]
            for seed in [101,202,303,404,505]:
                modified=[perturb(texts[i],kind,rate,seed,rows[i]['id']) if rows[i]['label'] else texts[i] for i in test]
                xp=v.transform(modified);s=model.predict_proba(xp)[:,1];p=s>=t;variants.append(p)
                normal_s=model.predict_proba(v.transform([normalise(z) for z in modified]))[:,1];normal_metrics.append(metrics(yt,normal_s>=t,normal_s))
                coverage.append(float(np.mean([modified[j]!=texts[i] for j,i in enumerate(test) if rows[i]['label']])))
                changed.append(float(np.mean(np.asarray((xp-xtest).getnnz(axis=1))[bec]>0)))
            recalls=[metrics(yt,p)['recall'] for p in variants];key=f'{kind}_{rate:.2f}'
            conditions[key]={'mean_recall':float(np.mean(recalls)),'delta_recall_percentage_points':100*(clean_metrics['recall']-float(np.mean(recalls))),'metrics_by_realisation':[metrics(yt,p) for p in variants],'mean_text_changed_fraction_bec':float(np.mean(coverage)),'mean_vector_changed_fraction_bec':float(np.mean(changed)),'normalisation_metrics_by_realisation':normal_metrics}
            predictions[key]=[p.astype(int).tolist() for p in variants]
            if rate==.1:conditions[key]['bootstrap']=bootstrap_lr(yt,clean,variants,gt)
    norm_scores=model.predict_proba(v.transform([normalise(texts[i]) for i in test]))[:,1]
    terms=v.get_feature_names_out();coef=model.coef_[0];order=np.argsort(coef)
    report={'stage':'primary_LR_exploratory_experiment','software':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},'input_selected_records':len(retained)+len(held)-sum(x['reason']=='cross_class_near_duplicate_group' for x in held),'screening_exclusions':dict(Counter(x['source']+':'+x['reason'] for x in held)),'retained_records':len(rows),'retained_by_class':dict(Counter(r['source'] for r in rows)),'near_duplicate_pairs_conservatively_grouped':len(pairs),'partitions':{n:{'messages':sum(r['split']==n for r in rows),'classes':dict(Counter(r['source'] for r in rows if r['split']==n)),'groups':len({r['group_id'] for r in rows if r['split']==n})} for n in ['train','validation','test']},'selected_configuration':{'C':c,'class_weight':weight,'threshold':t,'seed':42},'validation_trials':trials,'clean_test':clean_metrics,'normalised_clean_test':metrics(yt,norm_scores>=t,norm_scores),'conditions':conditions,'prevalence_precision':{str(pi):pi*clean_metrics['recall']/(pi*clean_metrics['recall']+(1-pi)*clean_metrics['fpr']) if pi*clean_metrics['recall']+(1-pi)*clean_metrics['fpr'] else None for pi in [.01,.1,.5]},'top_positive_features':[{'term':terms[i],'coefficient':float(coef[i])} for i in order[-30:][::-1]],'top_negative_features':[{'term':terms[i],'coefficient':float(coef[i])} for i in order[:30]],'elapsed_seconds':time.time()-start,'limitations':['Human quality-control audit incomplete; revised filtering is automated and exploratory.','Near-duplicate candidates automatically grouped conservatively without human relationship review.','Seven group-stratified folds give approximately 71/14/14, rather than exact 70/15/15.','Remaining personal, incomplete and out-of-scope records may persist.','Synthetic BEC versus historical human Enron source confounding remains.','Template families unavailable; similarity grouping cannot eliminate all leakage.','Only primary seed 42 has been run; other seeds, style audit and integrated gradients pending.','Confidence intervals condition on the fixed split and model; rare or zero false positives do not demonstrate deployment safety.']}
    (args.output/'lr_results_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.output/'lr_test_predictions.json').write_text(json.dumps({'ids':[rows[i]['id'] for i in test],'groups':gt.tolist(),'labels':yt.tolist(),'scores':scores.tolist(),'predictions':predictions}),encoding='utf-8')
    print(f'Finished. Clean test F1={clean_metrics["f1"]:.4f}; FPR={clean_metrics["fpr"]:.4f}',flush=True)
    print('Upload lr_results_summary.json only. Keep the pool, manifests, predictions and model local.',flush=True)

if __name__=='__main__':main()
