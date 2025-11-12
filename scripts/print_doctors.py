"""Admin helper: print doctor records for inspection.

Usage:
    python scripts/print_doctors.py --uri <MONGO_URI>

If no --uri is provided, the script uses MONGO_URI from the environment (.env).
The script prints: _id, name, email, specialization, embedding present (Y/N), and a 120-char profile snippet.
"""

import argparse
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', type=str, help='MongoDB URI (overrides MONGO_URI env)')
    args = parser.parse_args()

    mongo_uri = args.uri or os.getenv('MONGO_URI')
    if not mongo_uri:
        print('MONGO_URI not provided via --uri or environment')
        return

    client = MongoClient(mongo_uri)
    db = client['caremate']

    docs = list(db['users'].find({'role': 'doctor'}))
    if not docs:
        print('No doctors found in users collection')
        return

    specs_count = {}
    for d in docs:
        _id = d.get('_id')
        # Support multiple name representations
        name_obj = d.get('name')
        if isinstance(name_obj, dict):
            name = f"{name_obj.get('first','') or ''} {name_obj.get('last','') or ''}".strip() or name_obj.get('full','')
        else:
            name = f"{d.get('first','') or ''} {d.get('last','') or ''}".strip() or (d.get('name') or '<no name>')
        email = d.get('email','')
        profile = d.get('doctorProfile', {})
        specialization = (profile.get('specialization','') or '').strip()
        bio = profile.get('bio') or d.get('bio','') or ''
        qualifications = profile.get('qualifications', [])
        emb_present = 'Y' if d.get('embedding') else 'N'

        snippet_parts = []
        if specialization:
            snippet_parts.append(specialization)
        if qualifications:
            snippet_parts.append(', '.join(qualifications))
        if bio:
            snippet_parts.append(bio[:120])

        snippet = ' | '.join(snippet_parts)[:200]

        print(f"ID: {_id}\nName: {name}\nEmail: {email}\nSpecialization: {specialization}\nEmbedding: {emb_present}\nProfile: {snippet}\n---\n")

        # tally specializations
        key = specialization.lower() if specialization else '<unknown>'
        specs_count[key] = specs_count.get(key, 0) + 1

    # Summary
    print('\nSpecialization summary:')
    for spec, cnt in sorted(specs_count.items(), key=lambda x: -x[1]):
        print(f"  {spec}: {cnt}")


if __name__ == '__main__':
    main()
