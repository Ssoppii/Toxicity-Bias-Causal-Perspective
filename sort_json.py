import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Sort a JSON file by 'id' field")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., IMDB, XSum)")
    parser.add_argument("--topn", type=int, required=True, help="Top-N category")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save sorted output JSON file")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 정렬
    sorted_data = sorted(data, key=lambda x: int(x["id"]))

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    print(f"[{args.dataset}] top{args.topn}: Sorted {len(sorted_data)} samples → saved to {args.output_file}")

if __name__ == "__main__":
    main()