"""Export BEC review candidates without changing databases or making decisions."""
import argparse,csv,hashlib,json,sqlite3
from pathlib import Path

def safe_cell(value):
    value=str(value)
    return "'"+value if value.lstrip().startswith(('=','+','-','@')) else value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--eligibility',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in (a.preparation,a.eligibility,a.original):
        if not path.is_file():p.error('An input file was not found: '+str(path))
    prep=sqlite3.connect(a.preparation.resolve().as_uri()+'?mode=ro',uri=True)
    elig=sqlite3.connect(a.eligibility.resolve().as_uri()+'?mode=ro',uri=True)
    expected=dict(prep.execute('SELECT key,value FROM settings')).get('bec_sha256')
    if hashlib.sha256(a.original.read_bytes()).hexdigest()!=expected:p.error('Original CSV does not match the preparation source')
    with a.original.open(encoding='utf-8-sig',newline='') as f: originals=list(csv.DictReader(f))
    a.output.mkdir(parents=True,exist_ok=True)
    out=a.output/'bec_flagged_review.csv'
    if out.exists():p.error('Review sheet already exists. Keep existing decisions; use another output folder for a new export.')
    fields=['record_id','source_row_index','token_count','language_status','detected_language','language_score','review_flags','original_subject','original_body','cleaned_subject','cleaned_body','decision','reason','reviewer']
    counts={}; total=0
    with out.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for ident,n,status,lang,score in elig.execute("SELECT id,token_count,status,language,language_score FROM assessments WHERE source='synthetic_bec' AND status IN ('english_candidate','language_review','non_english_candidate') ORDER BY id"):
            record=prep.execute('SELECT local_key,subject,body,review_flags FROM records WHERE id=?',(ident,)).fetchone()
            if not record:raise RuntimeError('Record missing from preparation database')
            key,subject,body,flags=record
            if status=='english_candidate' and flags=='[]':continue
            original=originals[int(key)]
            row=dict(record_id=ident,source_row_index=key,token_count=n,language_status=status,detected_language=lang,language_score=round(score,6),review_flags=flags,original_subject=original['subject'],original_body=original['body'],cleaned_subject=subject,cleaned_body=body,decision='',reason='',reviewer='')
            w.writerow({k:safe_cell(v) for k,v in row.items()});total+=1;counts[status]=counts.get(status,0)+1
    summary={'stage':'flagged_bec_review_export_only','exported_records':total,'language_status_counts':counts,'decisions_made':0,'scope':'Flagged records and language-review records with nonblank bodies and at least 20 tokens; not the final random quality-control audit.'}
    (a.output/'bec_review_export_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.output/'REVIEW_INSTRUCTIONS.txt').write_text('Keep this sheet local. Original text is included to assess links before masking.\nFor each row fill decision with keep, exclude, or unsure; add reason and reviewer.\nkeep: English business social-engineering text fitting the text-only BEC scope, with usable content and no unresolved obfuscation.\nexclude: link/QR/attachment is the attack vehicle, non-English, automated/personal/ad text, unusable content, or confirmed obfuscation.\nunsure: cannot resolve language, attack vehicle, obfuscation, or content suitability.\nAn ordinary link or attachment mention alone is not an exclusion. Do not open links or attachments.\nInspect original and cleaned text for lost request sentences or failed redaction. Cleaning problems require resolution, not automatic approval.\nThis is targeted review, not the planned random quality-control audit.\nDo not edit identifiers or original/cleaned text columns. Save as CSV UTF-8.\n',encoding='utf-8')
    prep.close();elig.close();print(f'Finished. Exported {total} BEC records for review.');print('Upload bec_review_export_summary.json first. Keep the review CSV local for now.')

if __name__=='__main__':main()
