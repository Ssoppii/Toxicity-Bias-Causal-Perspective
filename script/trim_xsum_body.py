import os
import glob

TRIGGER_PHRASES = [
    "Copy this link",
    "These are external links and will open in a new window"
]

def extract_body_after_trigger(text: str) -> str:
    copy_link_index = text.find("Copy this link")
    second_trigger = "These are external links and will open in a new window"

    if copy_link_index != -1:
        sub_text = text[copy_link_index:]
        second_index = sub_text.find(second_trigger)
        if second_index != -1:
            return sub_text[second_index + len(second_trigger):].strip()
        else:
            # If only "Copy this link" is found, return everything after it
            return sub_text[len("Copy this link"):].strip()
    elif second_trigger in text:
        # If only "These are external links and will open in a new window" is found
        return text.split(second_trigger, 1)[-1].strip()
    else:
        # Fallback: return the full text
        return text.strip()

def process_file(file_path: str, output_dir: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # XSUM 태그 안에서만 처리
    if "[XSUM]RESTBODY[XSUM]" in content:
        parts = content.split("[XSUM]RESTBODY[XSUM]")
        header = parts[0]
        restbody = parts[1] if len(parts) > 1 else ""
        trimmed_body = extract_body_after_trigger(restbody)
        new_content = trimmed_body
    else:
        new_content = extract_body_after_trigger(content)

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(new_content)

def process_directory(input_dir: str, output_dir: str):
    files = glob.glob(os.path.join(input_dir, "*.data"))
    print(f"Found {len(files)} files to process.")
    for file_path in files:
        process_file(file_path, output_dir)
    print("Processing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .data files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trimmed files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)