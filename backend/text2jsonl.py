#!/usr/bin/env python3
# Convert the custom-formatted *.dat files to JSONL
import json
import sys
import os

def parse_custom_file(input_path):
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_entry = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip('\n')

        # Skip blank lines
        if line.strip() == '':
            if current_entry:
                entries.append(current_entry)
                current_entry = {}
            i += 1
            continue

        # Parse key:value lines except for yaml
        if line.startswith('yaml:'):
            # yaml key found; consume multiline value
            key = 'yaml'
            # Everything after 'yaml:' on the same line is part of the yaml (possibly nothing)
            yaml_value_lines = []
            after_colon = line[len('yaml:'):].lstrip()
            if after_colon:
                yaml_value_lines.append(after_colon)

            i += 1
            # Keep adding lines until blank line or EOF
            while i < n and lines[i].strip() != '':
                yaml_value_lines.append(lines[i].rstrip('\n'))
                i += 1

            current_entry[key] = '\n'.join(yaml_value_lines)
            continue

        # Regular key: value line
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            current_entry[key] = value

        i += 1

    # Add last entry if file doesn't end with blank line
    if current_entry:
        entries.append(current_entry)

    return entries

def write_jsonl(entries, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            f_out.write(json_line + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_filename>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}.jsonl"

    parsed_entries = parse_custom_file(input_file)
    write_jsonl(parsed_entries, output_file)
    print(f"Written {len(parsed_entries)} entries to {output_file}")
