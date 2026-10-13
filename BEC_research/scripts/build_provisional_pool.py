"""Build provisional exact/thread groups and matched pool. No splits or final labels."""
import argparse,csv,hashlib,json,random,re,sqlite3,time
from collections import Counter,defaultdict
from pathlib import Path

VERSION='1.0'
AUTO=re.compile(r'\b(?:delivery (?:status notification|failure)|undeliverable|out of office|automatic reply|system[- ]generated|automated (?:alert|message|notification)|calendar invitation|unsubscribe|remove me from (?:your|the) mailing list)\b',re.I)
AD=re.compile(r'\b(?:limited time offer|click here to unsubscribe|buy now.{0,30}(?:save|discount)|special offer.{0,30}expires)\b',re.I)

def bin_for(n):
    for edge,label in [(64,'20-63'),(128,'64-127'),(256,'128-255'),(512,'256-511')]:
        if n<edge:return label
    return '512+'

class Groups:
    def __init__(self):self.parent=[];self.size=[]
    def add(self):i=len(self.parent);self.parent.append(i);self.size.append(1);return i
    def root(self,i):
        while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def union(self,a,b):
        a,b=self.root(a),self.root(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b]

def screen(subject,body,flags):
    if not body.strip():return 'blank_body'
    if any(x in flags for x in ['format_character_review','mixed_script_review','parser_defect_review']):return 'unresolved_character_or_parser_review'
    if AUTO.search(subject+'\n'+body):return 'automated_notice_candidate'
    if AD.search(subject+'\n'+body):return 'advertisement_candidate'
    return ''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,required=True);p.add_argument('--eligibility',type=Path,required=True);p.add_argument('--bec-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for x in [a.preparation,a.eligibility,a.bec_manifest]:
        if not x.is_file():p.error('Missing input: '+str(x))
    a.output.mkdir(parents=True,exist_ok=True)
    dbpath=a.output/'provisional_pool.sqlite'
    if dbpath.exists():p.error('Output exists; use a new output folder to preserve previous work')
    prep=sqlite3.connect(a.preparation.resolve().as_uri()+'?mode=ro',uri=True)
    prep.execute('ATTACH DATABASE ? AS eligibility',(a.eligibility.resolve().as_uri()+'?mode=ro',))
    with a.bec_manifest.open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    allowed={r['record_id'] for r in manifest if r['curation_status']=='candidate_pending_quality_audit'}
    out=sqlite3.connect(dbpath)
    out.execute('CREATE TABLE candidates (id TEXT PRIMARY KEY,source TEXT,subject TEXT,body TEXT,token_count INTEGER,length_bin TEXT,group_id TEXT,selected INTEGER DEFAULT 0)')
    out.execute('CREATE TABLE exclusions (id TEXT PRIMARY KEY,source TEXT,reason TEXT)')
    uf=Groups();ids=[];origins=[];bodies={};mids={};links=[];status=Counter();seen_bec=set();start=time.monotonic()
    print('Screening stored candidate records; source emails will not be reread.',flush=True)
    q="SELECT r.id,r.source,r.subject,r.body,r.body_hash,r.message_id_hash,r.reference_hashes,r.review_flags,e.token_count,e.status FROM records r JOIN eligibility.assessments e ON e.id=r.id WHERE e.status='english_candidate' OR r.source='synthetic_bec' ORDER BY r.id"
    for count,(ident,source,subject,body,bh,mh,refs,flags,n,language) in enumerate(prep.execute(q),1):
        if source=='synthetic_bec':
            if ident not in allowed:continue
            seen_bec.add(ident)
        elif language!='english_candidate':continue
        reason=screen(subject,body,json.loads(flags))
        if reason:
            out.execute('INSERT INTO exclusions VALUES (?,?,?)',(ident,source,reason));status[source+':'+reason]+=1;continue
        i=uf.add();ids.append(ident);origins.append(source)
        out.execute('INSERT INTO candidates (id,source,subject,body,token_count,length_bin) VALUES (?,?,?,?,?,?)',(ident,source,subject,body,n,bin_for(n)))
        # Exact cleaned body across both classes; conflicts held later.
        if bh in bodies:uf.union(i,bodies[bh])
        else:bodies[bh]=i
        if mh:
            if mh in mids:uf.union(i,mids[mh])
            else:mids[mh]=i
        for ref in json.loads(refs):links.append((i,ref))
        status[source+':passed_rules']+=1
        if count%10000==0:out.commit();print(f'Scanned {count:,} records; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
    if seen_bec!=allowed:p.error('BEC manifest and candidate database disagree')
    print('Grouping exact bodies and explicit Message-ID/reply relationships...',flush=True)
    for i,ref in links:
        if ref in mids:uf.union(i,mids[ref])
    members=defaultdict(list)
    for i in range(len(ids)):members[uf.root(i)].append(i)
    groups_by_source=defaultdict(list);conflict=0
    for root,indices in members.items():
        sources={origins[i] for i in indices};group_id=min(ids[i] for i in indices)
        if len(sources)>1:
            conflict+=1
            for i in indices:out.execute('INSERT OR REPLACE INTO exclusions VALUES (?,?,?)',(ids[i],origins[i],'cross_source_group_conflict'));out.execute('DELETE FROM candidates WHERE id=?',(ids[i],))
            continue
        origin=next(iter(sources));group_ids=[ids[i] for i in indices]
        for ident in group_ids:out.execute('UPDATE candidates SET group_id=? WHERE id=?',(group_id,ident))
        groups_by_source[origin].append((group_id,group_ids))
    out.commit();out.execute('CREATE INDEX body_index ON candidates(source,body)');out.execute('CREATE INDEX group_index ON candidates(group_id)')
    # Deterministic representatives for exact cleaned-body copies, kept in same group.
    print('Retaining one representative per exact cleaned body...',flush=True)
    out.execute('CREATE TABLE representatives AS SELECT min(id) AS id FROM candidates GROUP BY source,body')
    out.execute('CREATE UNIQUE INDEX rep_id ON representatives(id)');out.commit()
    bec_ids=[x[0] for x in out.execute("SELECT c.id FROM candidates c JOIN representatives r ON r.id=c.id WHERE c.source='synthetic_bec' ORDER BY c.id")]
    for ident in bec_ids:out.execute('UPDATE candidates SET selected=1 WHERE id=?',(ident,))
    target=Counter(x[0] for x in out.execute("SELECT length_bin FROM candidates WHERE selected=1"))
    group_bins=defaultdict(Counter);group_reps=defaultdict(list)
    for ident,g,b in out.execute("SELECT c.id,c.group_id,c.length_bin FROM candidates c JOIN representatives r ON c.id=r.id WHERE c.source='enron' ORDER BY c.id"):
        group_bins[g][b]+=1;group_reps[g].append(ident)
    rng=random.Random(42);order=sorted(group_reps);rng.shuffle(order);selected=[];achieved=Counter()
    # Whole groups only, choosing groups that reduce absolute length-bin count error.
    for g in order:
        bins=group_bins[g]
        before=sum(abs(target[b]-achieved[b]) for b in set(target)|set(bins)|set(achieved))
        after=sum(abs(target[b]-(achieved[b]+bins[b])) for b in set(target)|set(bins)|set(achieved))
        if after<before:selected.append(g);achieved.update(bins)
    for g in selected:out.execute('UPDATE candidates SET selected=1 WHERE group_id=? AND id IN (SELECT id FROM representatives)',(g,))
    out.commit()
    fields=['record_id','source','group_id','token_count','length_bin']
    with (a.output/'provisional_selected_manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        w.writerows(out.execute('SELECT id,source,group_id,token_count,length_bin FROM candidates WHERE selected=1 ORDER BY source,id'))
    audit_ids=[]
    for source in ['synthetic_bec','enron']:
        available=[x[0] for x in out.execute('SELECT id FROM candidates WHERE source=? AND selected=1 ORDER BY id',(source,))]
        audit_ids.extend(random.Random(42).sample(available,min(200,len(available))))
    with (a.output/'quality_control_review.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['record_id','source','group_id','subject','body','suitability','reason','reviewer'])
        for ident in audit_ids:
            row=out.execute('SELECT id,source,group_id,subject,body FROM candidates WHERE id=?',(ident,)).fetchone()
            w.writerow([("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for v in row]+['','',''])
    summary={'stage':'provisional_matched_pool_no_train_test_splits','script_version':VERSION,'rule_counts':dict(status),'cross_source_conflicting_groups_held':conflict,'selected_records_by_source':dict(out.execute('SELECT source,count(*) FROM candidates WHERE selected=1 GROUP BY source')),'selected_groups_by_source':dict(out.execute('SELECT source,count(DISTINCT group_id) FROM candidates WHERE selected=1 GROUP BY source')),'target_bec_length_bins':dict(target),'achieved_enron_length_bins':dict(achieved),'quality_control_sample_size':len(audit_ids),'sampling_seed':42,'pending':['human quality-control review and redaction/completeness checks','near-duplicate candidate detection/review within selected pool','additional verified template-family grouping where available','freeze curation rules and grouped partitions'],'limitations':['BEC scope-held messages remain outside this provisional pool.','Screening rules provide suitability candidates, not verified fraud-negative labels.','Grouping currently covers exact cleaned bodies and explicit header reply links; subject similarity alone was not treated as a verified thread.','Near-duplicate review must precede train/validation/test splitting.','Names and other identifying information may remain; keep text/database local.']}
    (a.output/'provisional_pool_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    out.close();prep.close();print('Finished. Upload provisional_pool_summary.json only. Keep database and review CSV local.',flush=True)

if __name__=='__main__':main()
