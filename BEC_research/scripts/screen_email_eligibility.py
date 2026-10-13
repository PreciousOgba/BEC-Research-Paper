"""Resumable preliminary token/language checks; does not assign final labels."""
import argparse, hashlib, importlib.metadata, json, platform, re, sqlite3, time
from pathlib import Path

MODEL = 'distilbert/distilbert-base-uncased'
VERSION = '1.0'
CONFIDENCE = 0.90

def length_bin(n):
    if n < 20: return 'under_20'
    for bound, label in [(64,'20-63'),(128,'64-127'),(256,'128-255'),(512,'256-511')]:
        if n < bound: return label
    return '512+'

def language_check(text, detector, exception_type):
    # Masking markers are not natural language evidence.
    text = re.sub(r'\[(?:URL|EMAIL|PHONE|ACCOUNT)\]', ' ', text)
    try:
        result = detector(text)
    except exception_type:
        return '', 0.0, 'language_review'
    if not result: return '', 0.0, 'language_review'
    top = max(result, key=lambda x: x.prob)
    status = 'english_candidate' if top.lang == 'en' and top.prob >= CONFIDENCE else 'non_english_candidate' if top.lang != 'en' and top.prob >= CONFIDENCE else 'language_review'
    return top.lang, float(top.prob), status

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--database',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.database.is_file(): ap.error('Candidate database not found')
    from transformers import AutoTokenizer
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    from huggingface_hub import model_info
    DetectorFactory.seed=42
    args.output.mkdir(parents=True,exist_ok=True)
    source=sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True)
    total=source.execute('SELECT count(*) FROM records').fetchone()[0]
    settings={'script_version':VERSION,'model':MODEL,'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['transformers','tokenizers','huggingface-hub','langdetect']},'source_path':str(args.database.resolve()),'source_records':total,'source_size':args.database.stat().st_size,'source_mtime_ns':args.database.stat().st_mtime_ns,'source_settings':dict(source.execute('SELECT key,value FROM settings')),'minimum_tokens':20,'language_seed':42,'language_confidence_threshold':CONFIDENCE}
    config_path=args.output/'screening_config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding='utf-8')); comparable=dict(old); comparable.pop('checkpoint_revision',None)
        if comparable!=settings: ap.error('Input/environment changed; use a new output folder')
        revision=old['checkpoint_revision']
    else:
        print('Resolving tokenizer revision. Internet is needed for the initial download.',flush=True)
        revision=model_info(MODEL).sha
    print('Loading tokenizer only; no model training or model-weight download.',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=revision,use_fast=True)
    if not tokenizer.is_fast: ap.error('A fast tokenizer is required')
    tokenizer.save_pretrained(args.output/'tokenizer')
    settings['checkpoint_revision']=revision
    config_path.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    output=sqlite3.connect(args.output/'eligibility.sqlite')
    output.execute('CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY,value INTEGER)')
    output.execute('CREATE TABLE IF NOT EXISTS assessments (id TEXT PRIMARY KEY,source TEXT,token_count INTEGER,length_bin TEXT,language TEXT,language_score REAL,status TEXT,truncated_at_512 INTEGER,prior_flags TEXT)')
    output.execute('CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY,token_count INTEGER,language TEXT,language_score REAL,language_status TEXT)')
    output.commit()
    last=output.execute("SELECT value FROM progress WHERE key='last_rowid'").fetchone(); last=last[0] if last else 0
    start=time.monotonic(); new=0
    print(f'Starting/resuming. {output.execute("SELECT count(*) FROM assessments").fetchone()[0]:,} records already checked.',flush=True)
    cursor=source.execute('SELECT rowid,id,source,subject,body,review_flags FROM records WHERE rowid>? ORDER BY rowid',(last,))
    for rowid,ident,origin,subject,body,flags in cursor:
        text=subject+'\n'+body
        if not body.strip():
            n=0; lang=''; score=0.0; status='exclude_blank_body'
        else:
            key=hashlib.sha256(text.encode('utf-8')).hexdigest()
            cached=output.execute('SELECT token_count,language,language_score,language_status FROM cache WHERE hash=?',(key,)).fetchone()
            if cached: n,lang,score,status=cached
            else:
                n=len(tokenizer(text,add_special_tokens=False,truncation=False,return_attention_mask=False,return_token_type_ids=False,verbose=False)['input_ids'])
                if n<20: lang=''; score=0.0; status='exclude_under_20_tokens'
                else: lang,score,status=language_check(text,detect_langs,LangDetectException)
                output.execute('INSERT INTO cache VALUES (?,?,?,?,?)',(key,n,lang,score,status))
            if n<20: status='exclude_under_20_tokens'
        output.execute('INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?,?)',(ident,origin,n,length_bin(n),lang,score,status,int(n+tokenizer.num_special_tokens_to_add(pair=False)>512),flags))
        output.execute("INSERT OR REPLACE INTO progress VALUES ('last_rowid',?)",(rowid,))
        new+=1
        if new%1000==0:
            output.commit()
            print(f'Checked {output.execute("SELECT count(*) FROM assessments").fetchone()[0]:,}/{total:,}; elapsed {(time.monotonic()-start)/60:.1f} minutes',flush=True)
    output.commit()
    summary={'stage':'preliminary_eligibility_not_final_dataset','configuration':settings,'assessed_records':output.execute('SELECT count(*) FROM assessments').fetchone()[0],'counts_by_source_and_status':[{'source':s,'status':t,'count':n} for s,t,n in output.execute('SELECT source,status,count(*) FROM assessments GROUP BY source,status')],'english_candidate_length_bins':[{'source':s,'bin':b,'count':n} for s,b,n in output.execute("SELECT source,length_bin,count(*) FROM assessments WHERE status='english_candidate' GROUP BY source,length_bin")],'english_candidates_with_prior_review_flags':[{'source':s,'count':n} for s,n in output.execute("SELECT source,count(*) FROM assessments WHERE status='english_candidate' AND prior_flags!='[]' GROUP BY source")],'english_candidates_would_truncate':[{'source':s,'count':n} for s,n in output.execute("SELECT source,count(*) FROM assessments WHERE status='english_candidate' AND truncated_at_512=1 GROUP BY source")],'notes':['Language scores are heuristic detector outputs, not calibrated probabilities.','English candidates are not verified business messages or verified class labels.','Low-confidence language and prior flags still require review.','Grouping, duplicates, payload suitability, redaction audit, sampling, and quality-control review are pending.']}
    (args.output/'eligibility_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    output.close(); source.close()
    print('Finished. Upload eligibility_summary.json only. Keep databases local.',flush=True)

if __name__=='__main__': main()
