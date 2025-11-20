import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

#!/usr/bin/env python3
"""
xml_to_csv.py

Convert an input .xml file to a .csv file placed in an ./out folder (by default).
Usage:
    python xml_to_csv.py path/to/input.xml
    python xml_to_csv.py path/to/input.xml --outdir path/to/out
"""

import xml.etree.ElementTree as ET

def strip_ns(tag):
    # remove namespace if present
    if tag is None:
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def find_record_tag(root):
    # count tags across the tree (excluding the document root tag)
    counts = Counter()
    for el in root.iter():
        counts[strip_ns(el.tag)] += 1
    root_tag = strip_ns(root.tag)
    if counts:
        # prefer the most common tag that is not the root tag and occurs more than once
        candidates = [(t, c) for t, c in counts.items() if t != root_tag and c > 1]
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
    return None

def element_to_row(elem):
    # Convert an element into a flat dict:
    # - attributes become keys "@attrname"
    # - direct child element texts become keys "childtag"
    # - if element has text and no children, key "." stores text
    row = {}
    for k, v in elem.attrib.items():
        row[f"@{strip_ns(k)}"] = v
    children = list(elem)
    if children:
        for c in children:
            row[strip_ns(c.tag)] = (c.text or "").strip()
    else:
        row["."] = (elem.text or "").strip()
    return row

def xml_to_csv(input_path, outdir=None):
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "out")
    os.makedirs(outdir, exist_ok=True)

    tree = ET.parse(input_path)
    root = tree.getroot()

    record_tag = find_record_tag(root)
    if not record_tag:
        # fallback: use direct children of root as records if multiple
        direct_children = list(root)
        if len(direct_children) <= 1:
            raise ValueError("Could not determine repeating record elements in the XML.")
        records = direct_children
    else:
        # collect all elements with matching stripped tag name
        records = [el for el in root.iter() if strip_ns(el.tag) == record_tag]
        if not records:
            raise ValueError(f"No elements found with tag '{record_tag}'.")

    # Build rows and columns
    rows = []
    columns = set()
    for rec in records:
        r = element_to_row(rec)
        rows.append(r)
        columns.update(r.keys())

    # stable ordering: attributes first (sorted), then child tags sorted, then '.' if present
    cols = sorted([c for c in columns if c.startswith("@")]) + sorted([c for c in columns if not c.startswith("@") and c != "."])
    if "." in columns:
        cols.append(".")

    input_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(outdir, f"{input_name}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # ensure all keys exist
            row_filled = {k: r.get(k, "") for k in cols}
            writer.writerow(row_filled)

    return out_path

def main(argv):
    p = argparse.ArgumentParser(description="Convert an XML file to a CSV (outputs to ./out by default).")
    p.add_argument("input", help="Path to input .xml file")
    p.add_argument("--outdir", "-o", help="Output directory (default: ./out)", default=None)
    args = p.parse_args(argv)

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    try:
        out_path = xml_to_csv(input_path, args.outdir)
        print(out_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])