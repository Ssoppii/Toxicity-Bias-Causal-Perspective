import json
import os
import argparse


def combine_debiased(
    original_path: str,
    debiased_paths: dict,
    output_path: str
):
    """
    Combine multiple per-category debiased JSON outputs into a single JSON preserving only the specified category fields.

    Parameters:
    - original_path: path to the original train.json
    - debiased_paths: dict mapping category name to its debiased JSON file path
    - output_path: where to write the combined JSON
    """
    # Load original data
    with open(original_path, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)

    # Load each category's debiased records, indexed by id
    debiased_data = {}
    for cat, path in debiased_paths.items():
        print(cat, path)
        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        # Normalize to dict mapping id -> record
        if isinstance(records, dict):
            # records is a dict of id->record or other mapping
            id2rec = {
                str(rec.get('id')): rec
                for rec in records.values()
                if isinstance(rec, dict) and 'id' in rec
            }
        else:
            id2rec = {
                str(rec.get('id')): rec
                for rec in records
                if isinstance(rec, dict) and 'id' in rec
            }
        print(f"Category '{cat}' loaded {len(id2rec)} records. Sample IDs: {list(id2rec.keys())[:5]}")
        debiased_data[cat] = id2rec

    print(f"Original data has {len(orig_data)} records. Sample IDs: {[str(r.get('id')) for r in orig_data[:5]]}")
    
    # Build combined records
    combined = []
    for rec in orig_data:
        rec_id = str(rec.get('id'))
        new_rec = {
            'id': rec_id,
            'token': rec.get('token', [])
        }
        # For each category, copy over debiased tokens and ids
        for cat, mapping in debiased_data.items():
            deb_rec = mapping.get(rec_id)
            # print(rec_id)
            if deb_rec:
                new_rec[f'{cat}_tokens'] = deb_rec.get(f'{cat}_tokens', [])
                new_rec[f'{cat}_token_ids'] = deb_rec.get(f'{cat}_token_ids', [])
            else:
                new_rec[f'{cat}_tokens'] = []
                new_rec[f'{cat}_token_ids'] = []
        combined.append(new_rec)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write combined JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine per-category debiased JSONs into a single train_debiased.json"
    )
    parser.add_argument(
        '--original', type=str, required=True,
        help='Path to the original train.json'
    )
    parser.add_argument(
        '--output', type=str, default='output/train_debiased.json',
        help='Path to write combined JSON'
    )
    # four categories
    categories = ['toxicity', 'insult', 'threat', 'identity_attack']
    for cat in categories:
        parser.add_argument(
            f'--{cat}', type=str, required=True,
            help=f'Path to the {cat}-debiased JSON file'
        )
    args = parser.parse_args()

    debiased_paths = {cat: getattr(args, cat) for cat in categories}
    combine_debiased(
        original_path=args.original,
        debiased_paths=debiased_paths,
        output_path=args.output
    )
