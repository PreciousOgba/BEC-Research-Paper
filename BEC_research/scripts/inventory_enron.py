"""Read-only Enron inventory. Standard library only; does not assign labels."""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from email import policy
from email.parser import BytesHeaderParser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error('Source folder not found')
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = args.output / 'enron_inventory.csv'
    summary = args.output / 'enron_inventory_summary.json'
    if inventory.exists() or summary.exists():
        parser.error('Inventory outputs already exist; choose a new output folder')
    counts = Counter()
    mailboxes = Counter()
    start = time.monotonic()
    print('Reading email files. Progress appears every 5,000 files.', flush=True)
    with inventory.open('x', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['relative_path', 'file_sha256', 'bytes', 'has_subject', 'has_message_id', 'has_reply_reference', 'status'])
        writer.writeheader()
        def walk_error(error):
            counts['directory_errors'] += 1
            print('A directory could not be read; counted in summary.', flush=True)
        for folder, dirs, files in os.walk(args.source, onerror=walk_error):
            dirs.sort()
            for name in sorted(files):
                path = Path(folder) / name
                relative = path.relative_to(args.source)
                counts['files_seen'] += 1
                row = dict(relative_path=relative.as_posix(), file_sha256='', bytes='', has_subject='', has_message_id='', has_reply_reference='', status='read_error')
                try:
                    raw = path.read_bytes()
                    row.update(file_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                    msg = BytesHeaderParser(policy=policy.default).parsebytes(raw)
                    row.update(has_subject=bool(str(msg.get('Subject', '')).strip()), has_message_id=bool(msg.get('Message-ID')), has_reply_reference=bool(msg.get('References') or msg.get('In-Reply-To')), status='read_ok')
                    counts['files_read'] += 1
                    counts['empty_files'] += not bool(raw)
                    counts['missing_subjects'] += not row['has_subject']
                    counts['header_defects'] += bool(msg.defects)
                    mailboxes[relative.parts[0]] += 1
                except Exception as error:
                    row['status'] = type(error).__name__
                    counts['file_errors'] += 1
                writer.writerow(row)
                if counts['files_seen'] % 5000 == 0:
                    out.flush()
                    print(f"Processed {counts['files_seen']:,} files; elapsed {(time.monotonic()-start)/60:.1f} minutes", flush=True)
    result = {'stage': 'inventory_only_not_curated_or_labelled', 'counts': dict(counts), 'mailbox_count': len(mailboxes), 'files_per_mailbox': dict(sorted(mailboxes.items())), 'elapsed_seconds': round(time.monotonic()-start, 1)}
    summary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Finished: {counts['files_read']:,} files read; {counts['file_errors']} file errors.", flush=True)
    print(f'Summary: {summary}', flush=True)
    print('Share the summary JSON only. Keep the inventory CSV local.', flush=True)


if __name__ == '__main__':
    main()
