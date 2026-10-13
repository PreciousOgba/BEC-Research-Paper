"""Stage 1 extraction, not final curation: local SQLite storage; no dependencies."""
import argparse, csv, hashlib, json, os, re, sqlite3, time, unicodedata
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

VERSION = '1.0'
def digest(text):
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()

class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts=[]; self.hidden=0
    def handle_starttag(self, tag, attrs):
        if tag in ('script','style'): self.hidden += 1
        if tag in ('br','p','div','li'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script','style') and self.hidden: self.hidden -= 1
        if tag in ('p','div','li'): self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

def clean(text):
    # Conservative rules: ambiguous signatures/disclaimers require later review.
    lines=text.replace('\r\n','\n').replace('\r','\n').split('\n'); kept=[]
    for line in lines:
        if re.match(r'^\s*(-----Original Message-----|Begin forwarded message:|--\s*$)',line,re.I): break
        if re.match(r'^\s*On .{1,200}wrote:\s*$',line): break
        if not line.lstrip().startswith('>'): kept.append(line)
    return '\n'.join(kept)

URL=re.compile(r'\b(?:https?://|www\.)[^\s<>]+',re.I)
EMAIL=re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',re.I)
PHONE=re.compile(r'(?<!\w)(?:\+\d[\d ()-]{7,}\d|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4})(?!\w)')
ACCOUNT=re.compile(r'(?i)\b((?:bank\s+)?account\s*(?:number|no\.?|#)|routing\s*(?:number|no\.?|#))\s*[:=]?\s*[A-Z0-9-]{6,}')
def redact(text):
    text=URL.sub('[URL]',text); text=EMAIL.sub('[EMAIL]',text)
    text=ACCOUNT.sub(lambda m:m.group(1)+' [ACCOUNT]',text)
    return PHONE.sub('[PHONE]',text)

def make_record(source,key,subject,body,message_id='',refs='',defects=0):
    original=subject+'\n'+body
    flags=[]
    if not body.strip(): flags.append('blank_body')
    if URL.search(original): flags.append('link_requires_review_before_final_inclusion')
    if any(unicodedata.category(c)=='Cf' for c in original): flags.append('format_character_review')
    if re.search(r'[A-Za-z][\u0370-\u03ff\u0400-\u04ff]|[\u0370-\u03ff\u0400-\u04ff][A-Za-z]',original): flags.append('mixed_script_review')
    if defects: flags.append('parser_defect_review')
    line_breaks=clean(body).count('\n')
    subject=' '.join(redact(subject).split())
    body=' '.join(redact(clean(body)).split())
    return (digest(source+':'+key),source,key,subject,body,digest(message_id) if message_id else '',json.dumps([digest(x) for x in re.findall(r'<[^>]+>',refs)]),digest(body) if body else '',line_breaks,json.dumps(flags))

def extract_body(msg):
    if not msg.is_multipart():
        if msg.get_content_maintype()!='text': return ''
        text=msg.get_content()
        if msg.get_content_subtype()=='html':
            p=VisibleHTML(); p.feed(text); return ''.join(p.parts)
        return text
    parts=[p for p in msg.walk() if not p.is_multipart() and p.get_content_disposition()!='attachment']
    plain=[p for p in parts if p.get_content_type()=='text/plain']
    chosen=plain or [p for p in parts if p.get_content_type()=='text/html']
    result=[]
    for part in chosen:
        text=part.get_content()
        if part.get_content_type()=='text/html':
            p=VisibleHTML(); p.feed(text); text=''.join(p.parts)
        result.append(text)
    return '\n'.join(result)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--enron',type=Path,required=True); ap.add_argument('--bec',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.enron.is_dir() or not args.bec.is_file(): ap.error('Source folder or BEC CSV not found')
    args.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(args.output/'email_candidates.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT)')
    settings={'version':VERSION,'enron_path':str(args.enron.resolve()),'bec_sha256':hashlib.sha256(args.bec.read_bytes()).hexdigest()}
    old=dict(db.execute('SELECT key,value FROM settings'))
    if old and old!=settings: ap.error('Inputs or script version differ; use a new output folder')
    db.executemany('INSERT OR IGNORE INTO settings VALUES (?,?)',settings.items())
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,source TEXT,local_key TEXT,subject TEXT,body TEXT,message_id_hash TEXT,reference_hashes TEXT,body_hash TEXT,line_breaks INTEGER,review_flags TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS completed (key TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE IF NOT EXISTS errors (source TEXT,local_key TEXT,error_type TEXT)'); db.commit()
    insert='INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?)'
    done={x[0] for x in db.execute('SELECT key FROM completed')}; start=time.monotonic(); processed=0
    print('Starting extraction. Existing completed mailboxes will be skipped.',flush=True)
    if 'bec' not in done:
        with args.bec.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not {'subject','body','label'} <= set(reader.fieldnames or []): ap.error('BEC CSV fields do not match expected format')
            for i,row in enumerate(reader):
                if not any((v or '').strip() for v in row.values()): continue
                if row['label'] not in ('1','1.0'): ap.error('Unexpected BEC label')
                db.execute(insert,make_record('synthetic_bec',str(i),row['subject'] or '',row['body'] or ''))
        db.execute('INSERT INTO completed VALUES (?)',('bec',)); db.commit()
    for mailbox in sorted(args.enron.iterdir()):
        if not mailbox.is_dir() or 'enron:'+mailbox.name in done: continue
        print('Processing mailbox:',mailbox.name,flush=True)
        # One transaction per mailbox: interruption rolls back this mailbox only.
        for folder,dirs,files in os.walk(mailbox):
            dirs.sort()
            for name in sorted(files):
                path=Path(folder)/name; key=path.relative_to(args.enron).as_posix()
                try:
                    msg=BytesParser(policy=policy.default).parsebytes(path.read_bytes())
                    db.execute(insert,make_record('enron',key,str(msg.get('Subject','')),extract_body(msg),str(msg.get('Message-ID','')),str(msg.get('References',''))+' '+str(msg.get('In-Reply-To','')),len(msg.defects)))
                except Exception as e: db.execute('INSERT INTO errors VALUES (?,?,?)',('enron',key,type(e).__name__))
                processed+=1
                if processed%5000==0: print(f'Processed {processed:,} files this run; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
        db.execute('INSERT INTO completed VALUES (?)',('enron:'+mailbox.name,)); db.commit()
    summary={'stage':'extracted_candidates_not_final_labels_or_splits','script_version':VERSION,'records_by_source':dict(db.execute('SELECT source,count(*) FROM records GROUP BY source')),'blank_cleaned_bodies':dict(db.execute("SELECT source,count(*) FROM records WHERE body='' GROUP BY source")),'extraction_errors':db.execute('SELECT count(*) FROM errors').fetchone()[0],'records_with_review_flags':dict(db.execute("SELECT source,count(*) FROM records WHERE review_flags!='[]' GROUP BY source")),'completed_enron_mailboxes':db.execute("SELECT count(*) FROM completed WHERE key LIKE 'enron:%'").fetchone()[0],'pending':['language identification','DistilBERT token eligibility','business and payload suitability screening','redaction quality audit','review of flagged records','duplicate/thread grouping','length matching','quality-control audit','grouped splits']}
    (args.output/'preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    db.close(); print('Finished. Upload preparation_summary.json only; keep SQLite database local.',flush=True)

if __name__=='__main__': main()
