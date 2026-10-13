"""Reduced CPU DistilBERT experiment on the exact saved LR partitions.
Keep this file beside run_primary_lr.py. No test-driven tuning.
One prespecified learning rate (2e-5), primary seed 42, at most two epochs.
"""
print('Starting DistilBERT CPU experiment; loading libraries...',flush=True)
import argparse,gc,json,math,os,platform,random,re,string,time
from pathlib import Path
import numpy as np
import torch,transformers,sklearn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from run_primary_lr import metrics,threshold,perturb,normalise

MODEL='distilbert/distilbert-base-uncased'
REVISION='12040accade4e8a0f71eabdb258fecc2e7e948be'
SEEDS=[101,202,303,404,505]

def paired_intervals(y,lr,db,lv,dv,groups):
    _,inv=np.unique(groups,return_inverse=True);g=int(inv.max())+1
    def aggregate(p):
        return np.column_stack([np.bincount(inv,weights=z,minlength=g) for z in [(y==1)&p,(y==0)&p,(y==1)&~p,(y==0)&~p]])
    lc,dc=aggregate(lr),aggregate(db)
    ltp=np.column_stack([np.bincount(inv,weights=(y==1)&p,minlength=g) for p in lv])
    dtp=np.column_stack([np.bincount(inv,weights=(y==1)&p,minlength=g) for p in dv])
    def scores(c):
        a,b,f,n=c.T
        return np.column_stack([a/(a+b),a/(a+f),2*a/(2*a+b+f),b/(b+n)])
    rng=np.random.default_rng(606);chunks=[];redrawn=0
    while sum(len(x) for x in chunks)<10000:
        ix=rng.integers(0,g,(250,g));l=lc[ix].sum(1);d=dc[ix].sum(1)
        valid=(l[:,0]+l[:,2]>0)&(l[:,1]+l[:,3]>0)&(l[:,0]+l[:,1]>0)&(d[:,0]+d[:,1]>0)
        redrawn+=int((~valid).sum());ix=ix[valid];l=l[valid];d=d[valid]
        lm,dm=scores(l),scores(d);positives=l[:,0]+l[:,2]
        ldelta=(lm[:,1]-ltp[ix].sum(1).mean(1)/positives)*100
        ddelta=(dm[:,1]-dtp[ix].sum(1).mean(1)/positives)*100
        chunks.append(np.column_stack([dm-lm,ldelta,ddelta,ddelta-ldelta]))
    values=np.concatenate(chunks)[:10000]
    names=['precision_DB_minus_LR','recall_DB_minus_LR','f1_DB_minus_LR','fpr_DB_minus_LR','LR_delta_recall_percentage_points','DB_delta_recall_percentage_points','DB_minus_LR_delta_recall_percentage_points']
    return {'replicates':10000,'seed':606,'redrawn_undefined':redrawn,'percentile_95_ci':{k:np.percentile(values[:,i],[2.5,97.5]).tolist() for i,k in enumerate(names)}}

def style_features(text):
    words=re.findall(r'\b\w+\b',text);n=max(1,len(text))
    return [len(text),len(words),sum(map(len,words))/max(1,len(words)),sum(c in string.punctuation for c in text)/n,sum(c.isdigit() for c in text)/n,sum(c.isupper() for c in text)/n,sum(ord(c)>127 for c in text)/n,text.count('\n')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--epochs',type=int,default=2);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();pool=args.experiment/'experiment_pool.jsonl';predfile=args.experiment/'lr_test_predictions.json'
    if not pool.is_file() or not predfile.is_file():parser.error('Saved LR pool or predictions not found.')
    if args.epochs not in (1,2):parser.error('This reduced protocol supports one or two epochs.')
    if (args.output/'best_model').exists():parser.error('Output already contains a model; choose a different output folder.')
    args.output.mkdir(parents=True,exist_ok=True);start=time.time()
    torch.set_num_threads(max(1,args.threads));torch.set_num_interop_threads(1)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    rows=[json.loads(line) for line in pool.open(encoding='utf-8')]
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
    saved=json.loads(predfile.read_text());assert [r['id'] for r in test]==saved['ids'];assert [r['label'] for r in test]==saved['labels']
    print(f'Train {len(train)}; validation {len(val)}; test {len(test)}. Loading pretrained model (first download may take several minutes)...',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,num_labels=2).to('cpu')
    tokenizer.save_pretrained(args.output/'tokenizer')
    def encode(texts):return tokenizer(texts,truncation=True,max_length=512,padding=False)
    encoded=encode([r['text'] for r in train])
    records=[{**{k:encoded[k][i] for k in encoded},'labels':r['label']} for i,r in enumerate(train)]
    def collate(batch):return tokenizer.pad(batch,padding=True,return_tensors='pt')
    loader=DataLoader(records,batch_size=8,shuffle=True,collate_fn=collate,num_workers=0,generator=torch.Generator().manual_seed(42))
    parameters=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not (n.endswith('bias') or 'LayerNorm' in n or 'layer_norm' in n)],'weight_decay':.01},{'params':[p for n,p in model.named_parameters() if p.requires_grad and (n.endswith('bias') or 'LayerNorm' in n or 'layer_norm' in n)],'weight_decay':0.0}]
    optimizer=torch.optim.AdamW(parameters,lr=2e-5)
    updates=math.ceil(len(loader)/2)*args.epochs;scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=math.ceil(.1*updates),num_training_steps=updates)
    def predict(texts):
        model.eval();probs=[];loss_total=0
        with torch.no_grad():
            for j in range(0,len(texts),16):
                inputs=tokenizer(texts[j:j+16],truncation=True,max_length=512,padding=True,return_tensors='pt')
                logits=model(**inputs).logits
                probs.extend(torch.softmax(logits,dim=1)[:,1].cpu().numpy().tolist())
        return np.array(probs)
    vy=np.array([r['label'] for r in val]);best_loss=float('inf');logs=[];interrupted=False
    try:
        for epoch in range(args.epochs):
            model.train();optimizer.zero_grad(set_to_none=True);epoch_start=time.time();loss_sum=0;seen=0
            for step,batch in enumerate(loader):
                output=model(**batch);loss=output.loss;loss_sum+=loss.item()*len(batch['labels']);seen+=len(batch['labels']);(loss/2).backward()
                if (step+1)%2==0 or step+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                if (step+1)%10==0:
                    elapsed=time.time()-epoch_start;remaining=elapsed/(step+1)*(len(loader)-step-1)/60
                    print(f'Epoch {epoch+1}/{args.epochs}; batch {step+1}/{len(loader)}; estimated remaining in this epoch {remaining:.1f} minutes',flush=True)
            vs=predict([r['text'] for r in val]);v_loss=float(-np.mean(np.log(np.clip(np.where(vy==1,vs,1-vs),1e-12,1))))
            log={'epoch':epoch+1,'train_loss':loss_sum/seen,'validation_loss':v_loss,'elapsed_seconds':time.time()-epoch_start};logs.append(log)
            print(f'Epoch {epoch+1} complete; validation loss {v_loss:.5f}',flush=True)
            if v_loss<best_loss:
                best_loss=v_loss;model.save_pretrained(args.output/'best_model');(args.output/'training_progress.json').write_text(json.dumps({'logs':logs,'best_epoch':epoch+1},indent=2))
    except KeyboardInterrupt:
        interrupted=True
        if not (args.output/'best_model').exists():raise RuntimeError('No full epoch/checkpoint completed; no trained test results will be reported.')
        print('Training interrupted. Evaluating the best previously completed epoch.',flush=True)
    del optimizer,scheduler,parameters;model=None;gc.collect()
    model=AutoModelForSequenceClassification.from_pretrained(args.output/'best_model').to('cpu')
    vs=predict([r['text'] for r in val]);t,vm=threshold(vy,vs)
    (args.output/'selected_threshold.json').write_text(json.dumps({'threshold':t,'validation':vm,'selected_before_test':True}))
    print(f'Validation threshold fixed at {t:.2f}. Evaluating held-out test messages...',flush=True)
    yt=np.array(saved['labels']);groups=np.array(saved['groups']);clean_scores=predict([r['text'] for r in test]);clean=clean_scores>=t;lr=np.array(saved['predictions']['clean'],dtype=bool)
    clean_metrics=metrics(yt,clean,clean_scores);conditions={};predictions={'clean':clean.astype(int).tolist()};normalised_cache={}
    original_ids=encode([r['text'] for r in test])['input_ids'];is_bec=yt==1
    original_lengths=[len(tokenizer(r['text'],truncation=False,add_special_tokens=True)['input_ids']) for r in test]
    for kind in ['homoglyph','zero_width']:
        key=f'{kind}_0.10';variants=[];norm=[];checks=[]
        for seed in SEEDS:
            print(f'Evaluating {kind}, realisation seed {seed}...',flush=True)
            texts=[perturb(r['text'],kind,.1,seed,r['id']) if r['label'] else r['text'] for r in test]
            ids=encode(texts)['input_ids'];changed=[i for i in range(len(test)) if ids[i]!=original_ids[i]]
            scores=clean_scores.copy()
            if changed:scores[changed]=predict([texts[i] for i in changed])
            p=scores>=t;variants.append(p)
            checks.append({'seed':seed,'token_ids_changed_fraction_bec':float(np.mean([ids[i]!=original_ids[i] for i in range(len(test)) if is_bec[i]])),'text_changed_fraction_bec':float(np.mean([texts[i]!=test[i]['text'] for i in range(len(test)) if is_bec[i]])),'truncated_count':sum(len(tokenizer(s,truncation=False,add_special_tokens=True)['input_ids'])>512 for s in texts)})
            # Deduplicate identical normalised inputs; especially useful when zero-width insertion is erased.
            norm_texts=[normalise(x) for x in texts];unique=[x for x in dict.fromkeys(norm_texts) if x not in normalised_cache]
            if unique:normalised_cache.update(zip(unique,predict(unique)))
            ns=np.array([normalised_cache[x] for x in norm_texts]);norm.append(metrics(yt,ns>=t,ns))
        lv=[np.array(p,dtype=bool) for p in saved['predictions'][key]]
        ms=[metrics(yt,p) for p in variants];delta=100*(clean_metrics['recall']-np.mean([m['recall'] for m in ms]));lrdelta=100*(metrics(yt,lr)['recall']-np.mean([metrics(yt,p)['recall'] for p in lv]))
        conditions[key]={'metrics_by_realisation':ms,'mean_recall':float(np.mean([m['recall'] for m in ms])),'delta_recall_percentage_points':float(delta),'DB_minus_LR_delta_recall_percentage_points':float(delta-lrdelta),'paired_group_bootstrap':paired_intervals(yt,lr,clean,lv,variants,groups),'normalisation_metrics_by_realisation':norm,'input_checks':checks}
        predictions[key]=[p.astype(int).tolist() for p in variants]
    print('Running the surface-style probe...',flush=True)
    sf=np.array([style_features(r['text']) for r in rows]);ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);si=np.array([i for i,r in enumerate(rows) if r['split']=='test']);labels=np.array([r['label'] for r in rows]);scaler=StandardScaler().fit(sf[ti]);probe=LogisticRegression(C=1,solver='liblinear',max_iter=1000,random_state=42).fit(scaler.transform(sf[ti]),labels[ti]);pt,pvm=threshold(labels[vi],probe.predict_proba(scaler.transform(sf[vi]))[:,1]);ps=probe.predict_proba(scaler.transform(sf[si]))[:,1]
    best_epoch=min(logs,key=lambda z:z['validation_loss'])['epoch']
    report={'stage':'reduced_CPU_DistilBERT_primary_experiment','software':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__},'device':'CPU','threads':args.threads,'precision':'float32','checkpoint':MODEL,'checkpoint_revision':REVISION,'learning_rate':2e-5,'effective_batch_size':16,'physical_batch_size':8,'training_seed':42,'planned_max_epochs':args.epochs,'completed_epochs':len(logs),'selected_epoch':best_epoch,'training_interrupted':interrupted,'training_logs':logs,'threshold':t,'validation':vm,'clean_test':clean_metrics,'test_counts':{'BEC':int(sum(yt==1)),'Enron':int(sum(yt==0))},'original_truncated_count':sum(n>512 for n in original_lengths),'conditions':conditions,'style_probe':{'features':['character_count','word_count','mean_word_length','punctuation_fraction','digit_fraction','uppercase_fraction','non_ASCII_fraction','line_break_count'],'threshold':pt,'validation':pvm,'clean_test':metrics(labels[si],ps>=pt,ps)},'elapsed_seconds':time.time()-start,'protocol_deviations':['One fixed learning rate (2e-5) instead of the three-rate search; CPU deadline constraint.','At most two epochs rather than four; only primary training seed 42.','Only rate 0.10 perturbations evaluated for DistilBERT; rates 0.05 and 0.20 remain LR-only sensitivity results.','Full human quality-control audit and manually reviewed near-duplicate relationships incomplete.','Integrated gradients, held-out styles and manual error audit not completed.'],'limitations':['Scores compare synthetic BEC with historical Enron correspondence; source confounding remains.','The LR model was tuned over eight configurations; DistilBERT uses one prespecified configuration, so tuning budgets differ.','Zero observed false positives in a small test set does not establish a zero deployment FPR.','Bootstrap intervals condition on the fixed split and fitted models; no equivalence claim.']}
    (args.output/'distilbert_results_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(args.output/'distilbert_test_predictions.json').write_text(json.dumps({'ids':saved['ids'],'groups':saved['groups'],'labels':saved['labels'],'scores':clean_scores.tolist(),'predictions':predictions}))
    print(f'Finished. DistilBERT test F1={clean_metrics["f1"]:.4f}; FPR={clean_metrics["fpr"]:.4f}',flush=True)
    print('Upload distilbert_results_summary.json only. Keep model and private data local.',flush=True)

if __name__=='__main__':main()
