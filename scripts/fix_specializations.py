"""Fix common specialization typos in doctor profiles.

Usage:
    # Dry run (no writes)
    python scripts/fix_specializations.py --uri "<MONGO_URI>"

    # Apply changes
    python scripts/fix_specializations.py --uri "<MONGO_URI>" --apply

The script finds doctors in `users` collection and normalizes
`doctorProfile.specialization` according to a mapping.
"""

import argparse
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

CANONICAL_MAP = {
    # lowercase source -> canonical Titlecase
    "pediadrist": "Pediatrician",
    "pediatrist": "Pediatrician",
    "pediatrician": "Pediatrician",
    "cardiologist": "Cardiologist",
    "cardiology": "Cardiologist",
    "ent": "ENT",
    "ear nose throat": "ENT",
    "neurology": "Neurology",
    "neuro": "Neurology",
    "dermatologist": "Dermatologist",
    "derm": "Dermatologist",
    "orthopedist": "Orthopedist",
    "orthopedic": "Orthopedist",
}


def canonicalize(spec):
    if not spec:
        return spec
    s = spec.strip().lower()
    return CANONICAL_MAP.get(s, spec.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', type=str, help='MongoDB URI (overrides MONGO_URI env)')
    parser.add_argument('--apply', action='store_true', help='Apply changes to the DB')
    args = parser.parse_args()

    mongo_uri = args.uri or os.getenv('MONGO_URI')
    if not mongo_uri:
        print('MONGO_URI not provided via --uri or environment')
        return

    client = MongoClient(mongo_uri)
    db = client['caremate']

    doctors = list(db['users'].find({'role': 'doctor'}))
    if not doctors:
        print('No doctors found')
        return

    changes = []
    for d in doctors:
        _id = d.get('_id')
        profile = d.get('doctorProfile', {})
        raw = profile.get('specialization') if profile is not None else None
        if raw is None:
            continue
        new = canonicalize(raw)
        if new != raw:
            changes.append((_id, raw, new))

    if not changes:
        print('No specialization typos found matching the canonical map.')
        return

    print(f'Found {len(changes)} records to change:')
    for _id, old, new in changes:
        print(f'  {_id}: "{old}" -> "{new}"')

    if args.apply:
        for _id, old, new in changes:
            result = db['users'].update_one({'_id': _id}, {'$set': {'doctorProfile.specialization': new}})
            print(f'Updated {_id}: modified_count={result.modified_count}')
        print('All updates applied.')
    else:
        print('\nDry run only. To apply these changes run the script with --apply')


if __name__ == '__main__':
    main()
