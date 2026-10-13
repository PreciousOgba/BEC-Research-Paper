"""Provisional BEC curation and duplicate candidates; no training/test splits."""
import argparse,csv,json,math,re,sqlite3,zipfile
from collections import Counter,defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def read_review(path):
    with zipfile.ZipFile(path) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
        def value(c):
            if c.get('t')=='inlineStr':return ''.join(c.find('s:is',NS).itertext())
            v=c.find('s:v',NS)
            return '' if v is None else strings[int(v.text)] if c.get('t')=='s' else v.text
        root=ET.fromstring(z.read('xl/worksheets/sheet1.xml')); results={}
        for row in root.findall('s:sheetData/s:row',NS):
            cells={re.sub(r'\d','',c.get('r')):value(c) for c in row.findall('s:c',NS)}
            ident=cells.get('H','')
            if not re.fullmatch('[a-f0-9]{64}',ident):continue
            if ident in results:raise ValueError('Duplicate review identifier')
            suggestion=cells.get('B',''); final=cells.get('D','')
            if suggestion not in ('keep','exclude','unsure') or final not in ('','keep','exclude','unsure'):raise ValueError('Invalid review decision')
            if final and not cells.get('E','').strip():raise ValueError('Human decision has no reviewer')
            results[ident]={'suggestion':suggestion,'suggestion_reason':cells.get('C',''),'final':final,'human_reviewer':cells.get('E',''),'human_notes':cells.get('F','')}
        if len(results)!=112:raise ValueError('Expected all 112 review records; workbook structure may have changed')
        return results

def grams(text):
    # Separate from predictive features. Whitespace-normalised, case-preserving body.
    text=' '.join(text.split())
    return {text[i:i+5] for i in range(max(0,len(text)-4))}

def near_pairs(bodies):
    sets={ident:grams(body) for ident,body in bodies.items()}
    frequency=Counter(g for s in sets.values() for g in s)
    index=defaultdict(list)
    for number,(ident,s) in enumerate(sorted(sets.items()),1):
        if not s:continue
        ordered=sorted(s,key=lambda g:(frequency[g],g))
        prefix=ordered[:len(s)-math.ceil(.90*len(s))+1]
        candidates={other for g in prefix for other in index[g]}
        for other in sorted(candidates):
            t=sets[other]
            if min(len(s),len(t))/max(len(s),len(t))<.90:continue
            intersection=len(s&t); union=len(s)+len(t)-intersection
            if 10*intersection>=9*union:yield other,ident,intersection/union
        for g in prefix:index[g].append(ident)
        if number%500==0:print(f'Near-duplicate check: {number:,} distinct bodies',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',type=Path,required=True);p.add_argument('--eligibility',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for x in (a.preparation,a.eligibility,a.review):
        if not x.is_file():p.error('Input not found: '+str(x))
    a.output.mkdir(parents=True,exist_ok=True)
    outputs=['bec_curation_manifest.csv','bec_exact_duplicate_members.csv','bec_near_duplicate_review.csv','bec_curation_summary.json']
    if any((a.output/x).exists() for x in outputs):p.error('Output files already exist; choose a new output folder to preserve prior work')
    reviews=read_review(a.review)
    prep=sqlite3.connect(a.preparation.resolve().as_uri()+'?mode=ro',uri=True)
    elig=sqlite3.connect(a.eligibility.resolve().as_uri()+'?mode=ro',uri=True)
    records=[]; seen=set(); status_counts=Counter(); provenance=Counter(); groups=defaultdict(list); triage=Counter()
    print('Applying review decisions to BEC candidates only...',flush=True)
    for ident,n,status in elig.execute("SELECT id,token_count,status FROM assessments WHERE source='synthetic_bec' ORDER BY id"):
        row=prep.execute('SELECT body,body_hash,review_flags FROM records WHERE id=?',(ident,)).fetchone()
        if row is None:raise ValueError('Preparation/eligibility identifiers disagree')
        body,body_hash,flags=row; review=reviews.get(ident);reason='';origin='automated_preliminary_screen';reviewer='';suggestion='';notes=''
        if review:
            seen.add(ident);suggestion=review['suggestion']
            if review['final']:
                origin='human_confirmed_AI_assisted';reviewer=review['human_reviewer'];notes=review['human_notes']
                state={'keep':'candidate_pending_quality_audit','exclude':'excluded_by_human','unsure':'held_uncertain'}[review['final']]
                reason=notes or review['suggestion_reason']
            else:
                origin='AI_suggestion_not_human_reviewed';reason=review['suggestion_reason']
                state='withheld_AI_exclusion' if suggestion=='exclude' else 'held_pending_human_review'
        elif status=='english_candidate':state='candidate_pending_quality_audit'
        elif status in ('exclude_blank_body','exclude_under_20_tokens'):state=status
        else:state='held_language_review'
        pending=[]
        # Flag broader payload/placeholder scenarios missed by URL-only screening.
        if re.search(r'\b(scan.{0,40}QR|QR.{0,30}(code|portal)|MALICIOUS LINK)\b',body,re.I):pending.append('qr_or_link_placeholder_review')
        if re.search(r'\b(attached|attachment|DocuSign|portal|click here|download)\b',body,re.I):pending.append('possible_payload_or_portal_review')
        if re.search(r'\b(From:|Org:|Amt:)\b',body):pending.append('generation_metadata_review')
        if flags!='[]' and not review:pending.append('unresolved_prior_flags')
        if pending and state=='candidate_pending_quality_audit':state='held_additional_scope_review'
        if state=='candidate_pending_quality_audit':groups[body_hash].append((ident,body))
        triage.update(pending);status_counts[state]+=1;provenance[origin]+=1
        records.append(dict(record_id=ident,token_count=n,language_status=status,curation_status=state,decision_provenance=origin,reviewer=reviewer,ai_suggestion=suggestion,reason=reason,scope_review_flags=json.dumps(pending),exact_body_group=body_hash))
    if seen!=set(reviews):raise ValueError('Review identifiers not found in source BEC records')
    def write(name,fields,rows):
        with (a.output/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(outputs[0],list(records[0]),records)
    representatives={};member_rows=[]
    for h,members in sorted(groups.items()):
        members.sort();rep=members[0][0];representatives[rep]=members[0][1]
        for ident,body in members:member_rows.append(dict(record_id=ident,representative_id=rep,exact_body_group=h))
    write(outputs[1],['record_id','representative_id','exact_body_group'],member_rows)
    pair_count=0
    with (a.output/outputs[2]).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['record_id_a','record_id_b','jaccard','relationship_decision','reviewer','notes']);w.writeheader()
        for first,second,score in near_pairs(representatives):
            w.writerow(dict(record_id_a=first,record_id_b=second,jaccard=score,relationship_decision='',reviewer='',notes=''));pair_count+=1
    summary={'stage':'provisional_BEC_curation_no_final_splits','total_bec_assessed':len(records),'curation_status_counts':dict(status_counts),'decision_provenance_counts':dict(provenance),'scope_flag_counts_overlapping':dict(triage),'candidate_records_before_exact_dedup':sum(len(x) for x in groups.values()),'distinct_candidate_bodies':len(representatives),'near_duplicate_pairs_needing_review':pair_count,'near_duplicate_definition':'Exact set Jaccard >=0.90 on character five-grams of whitespace-normalised case-preserving cleaned body; prefix filtering then exact verification.','pending':['additional scope review','confirmation/documentation of AI exclusions','review of near-duplicate relationships','Enron screening and grouping','matched sampling and random human quality-control audit','grouped splits'],'notes':['No source database was modified.','AI exclusions are withheld provisionally, not labelled human-reviewed.','Candidate retention is not independent verification of fraudulent intent.','This does not complete curation or form train/test partitions.']}
    (a.output/outputs[3]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    prep.close();elig.close();print('Finished. Upload bec_curation_summary.json only; keep manifests locally.',flush=True)

if __name__=='__main__':main()
