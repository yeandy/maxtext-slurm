#!/usr/bin/env python3
"""
Merge multiple trace.json.gz files into a single timeline-aligned trace.
Usage: merge_xplane_traces.py <path1> [path2] [path3] ...

Paths can be:
- Directories: all .trace.json.gz and .trace.json files in the directory will be merged
- Files: specific .trace.json.gz or .trace.json files
- Wildcards: e.g., ./traces/*.trace.json.gz

The output merged.trace.json.gz will be written to the current working directory.
"""

import json
import gzip
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import glob
import argparse


def load_trace(filepath: str) -> Dict[str, Any]:
    """Load a trace JSON file (supports both .json and .json.gz)."""
    print(f"Loading {filepath}...")
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def parse_id_value(val):
    """Parse an ID value which can be numeric or string (including hex)."""
    if isinstance(val, str):
        try:
            return int(val, 16) if val.lower().startswith('0x') else int(val)
        except ValueError:
            return 0
    return val if isinstance(val, int) else 0


def offset_id_value(val, offset):
    """Apply offset to an ID value, preserving its format."""
    if isinstance(val, str):
        try:
            if val.lower().startswith('0x'):
                return hex(int(val, 16) + offset)
            else:
                return str(int(val) + offset)
        except ValueError:
            return val
    return val + offset


def extract_process_type(process_name: str) -> str:
    """Extract the type from a process name matching pattern (type)[: ]?.*"""
    for delimiter in [':', ' ']:
        if delimiter in process_name:
            return process_name.split(delimiter, 1)[0]
    return process_name


def collect_process_info(traces: List[Tuple[str, Dict[str, Any]]]) -> Tuple[Dict[Tuple[int, int], str], Set[str]]:
    """Collect process types and PIDs from all traces."""
    pid_to_process_type = {}  # (node_id, original_pid) -> process_type
    process_types = set()

    for node_id, (filepath, trace) in enumerate(traces):
        if 'traceEvents' not in trace:
            continue

        for event in trace['traceEvents']:
            if event.get('ph') == 'M' and event.get('name') == 'process_name':
                pid = event.get('pid')
                if pid is not None and 'args' in event and 'name' in event['args']:
                    process_name = event['args']['name']
                    process_type = extract_process_type(process_name)
                    pid_to_process_type[(node_id, pid)] = process_type
                    process_types.add(process_type)

    return pid_to_process_type, process_types


def build_global_pid_map(traces: List[Tuple[str, Dict[str, Any]]],
                         pid_to_process_type: Dict[Tuple[int, int], str],
                         type_to_index: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    """Build global PID mapping ordered by (type, node_id, original_pid)."""
    all_pids = []

    for node_id, (filepath, trace) in enumerate(traces):
        if 'traceEvents' not in trace:
            continue

        # Collect unique PIDs for this node
        node_pids = set()
        for event in trace['traceEvents']:
            if 'pid' in event:
                node_pids.add(event['pid'])

        # Add to global list with type info
        for original_pid in node_pids:
            process_type = pid_to_process_type.get((node_id, original_pid), '')
            all_pids.append((process_type, node_id, original_pid))

    # Sort by (type, node_id, original_pid) and assign contiguous PIDs
    all_pids.sort(key=lambda x: (type_to_index.get(x[0], 999), x[1], x[2]))

    global_pid_map = {}
    for new_pid, (process_type, node_id, original_pid) in enumerate(all_pids):
        global_pid_map[(node_id, original_pid)] = new_pid

    return global_pid_map


def calculate_flow_id_offsets(traces: List[Tuple[str, Dict[str, Any]]]) -> List[int]:
    """Calculate flow ID offsets for each node."""
    max_flow_ids = []

    for filepath, trace in traces:
        max_flow_id = 0

        if 'traceEvents' in trace:
            for event in trace['traceEvents']:
                # Check all flow ID fields
                for key in ['id', 'bind_id']:
                    if key in event:
                        max_flow_id = max(max_flow_id, parse_id_value(event[key]))

                # Handle id2 (can be dict or value)
                if 'id2' in event:
                    if isinstance(event['id2'], dict):
                        for subkey in ['local', 'global']:
                            if subkey in event['id2']:
                                max_flow_id = max(max_flow_id, parse_id_value(event['id2'][subkey]))
                    else:
                        max_flow_id = max(max_flow_id, parse_id_value(event['id2']))

        max_flow_ids.append(max_flow_id)

    # Calculate cumulative offsets
    offsets = [0]
    for i in range(1, len(traces)):
        offsets.append(offsets[i-1] + max_flow_ids[i-1] + 1)

    return offsets


def remap_event_ids(event: Dict[str, Any], node_id: int,
                    global_pid_map: Dict[Tuple[int, int], int],
                    flow_id_offset: int) -> Dict[str, Any]:
    """Remap all IDs in an event to global space."""
    new_event = event.copy()

    # Remap PID
    if 'pid' in new_event:
        original_pid = new_event['pid']
        new_event['pid'] = global_pid_map.get((node_id, original_pid), original_pid)

    # Offset flow IDs
    if 'id' in new_event:
        new_event['id'] = offset_id_value(new_event['id'], flow_id_offset)

    if 'id2' in new_event:
        if isinstance(new_event['id2'], dict):
            new_id2 = new_event['id2'].copy()
            for key in ['local', 'global']:
                if key in new_id2:
                    new_id2[key] = offset_id_value(new_id2[key], flow_id_offset)
            new_event['id2'] = new_id2
        else:
            new_event['id2'] = offset_id_value(new_event['id2'], flow_id_offset)

    if 'bind_id' in new_event:
        new_event['bind_id'] = offset_id_value(new_event['bind_id'], flow_id_offset)

    # Remap PID in args
    if 'args' in new_event and isinstance(new_event['args'], dict):
        new_event['args'] = new_event['args'].copy()
        if 'pid' in new_event['args']:
            original_arg_pid = new_event['args']['pid']
            new_event['args']['pid'] = global_pid_map.get((node_id, original_arg_pid), original_arg_pid)

    return new_event


def process_metadata_event(event: Dict[str, Any], node_id: int,
                           node_prefix: str, node_name: str) -> Dict[str, Any]:
    """Process metadata events (process_name, process_sort_index)."""
    if event.get('name') == 'process_name':
        if 'args' not in event:
            event['args'] = {}
        event_name = event['args'].get('name', '')
        delimiter = '' if event_name.startswith('/') else '/'
        event['args']['name'] = f"{node_prefix}{delimiter}{event_name}"
        event['args']['node_id'] = node_id
        event['args']['node_name'] = node_name

    elif event.get('name') == 'process_sort_index':
        if 'args' not in event:
            event['args'] = {}
        event['args']['sort_index'] = event['pid']

    return event


def merge_xplane_traces(trace_files: List[str], output_path: str):
    """Merge multiple trace files into one, preserving original timestamps."""
    if not trace_files:
        print("Error: No input files provided")
        sys.exit(1)

    # Load all traces
    traces = []
    for filepath in trace_files:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        try:
            traces.append((filepath, load_trace(filepath)))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not traces:
        print("Error: No valid traces loaded")
        sys.exit(1)

    print(f"\nLoaded {len(traces)} traces")

    # Calculate node prefix width
    num_nodes = len(traces)
    id_width = len(str(num_nodes - 1))

    # Collect process information
    pid_to_process_type, process_types = collect_process_info(traces)

    # Create sorted type mapping
    sorted_types = sorted(process_types)
    type_to_index = {ptype: idx for idx, ptype in enumerate(sorted_types)}

    print(f"\nFound {len(sorted_types)} process types:")
    for ptype, type_idx in sorted(type_to_index.items(), key=lambda x: x[1]):
        print(f"  {ptype}: type_index={type_idx}")

    # Build global PID mapping
    global_pid_map = build_global_pid_map(traces, pid_to_process_type, type_to_index)
    print(f"\nTotal processes: {len(global_pid_map)}")

    # Calculate flow ID offsets
    flow_id_offsets = calculate_flow_id_offsets(traces)
    print(f"\nUsing flow ID offsets: {flow_id_offsets}")

    # Prepare merged trace
    merged = {
        'displayTimeUnit': traces[0][1].get('displayTimeUnit', 'ns'),
        'traceEvents': [],
    }
    for key in traces[0][1]:
        if key not in ['traceEvents', 'displayTimeUnit']:
            merged[key] = traces[0][1][key]

    # Process each trace
    metadata_events = []
    regular_events = []

    for node_id, (filepath, trace) in enumerate(traces):
        node_name = Path(filepath).stem.replace('.trace.json', '')
        node_prefix = f"n-{node_id:0{id_width}d}"
        flow_id_offset = flow_id_offsets[node_id]

        print(f"Processing {Path(filepath).name} (node_id={node_id}, prefix={node_prefix})")

        if 'traceEvents' not in trace:
            continue

        for event in trace['traceEvents']:
            if not event:
                continue

            # Remap all IDs
            new_event = remap_event_ids(event, node_id, global_pid_map, flow_id_offset)

            # Handle metadata events
            if new_event.get('ph') == 'M':
                new_event = process_metadata_event(new_event, node_id, node_prefix, node_name)
                metadata_events.append(new_event)
            else:
                regular_events.append(new_event)

    # Combine metadata and regular events
    merged['traceEvents'] = metadata_events + regular_events

    print(f"\nMerged {len(merged['traceEvents'])} total events")

    # Write output
    print(f"Writing to {output_path}...")
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(merged, f)

    print(f"Done! Merged trace saved to {output_path}")
    print(f"You can view it in Chrome at chrome://tracing or with Perfetto UI")


def collect_trace_files(paths: List[str]) -> List[str]:
    """Collect all trace files from the given paths."""
    all_files = []

    for path in paths:
        expanded = glob.glob(path)

        if not expanded:
            if os.path.isdir(path):
                all_files.extend(glob.glob(os.path.join(path, "*.trace.json.gz")))
                all_files.extend(glob.glob(os.path.join(path, "*.trace.json")))
            elif os.path.isfile(path):
                all_files.append(path)
            else:
                print(f"Warning: Path not found: {path}")
        else:
            for item in expanded:
                if os.path.isdir(item):
                    all_files.extend(glob.glob(os.path.join(item, "*.trace.json.gz")))
                    all_files.extend(glob.glob(os.path.join(item, "*.trace.json")))
                elif os.path.isfile(item):
                    all_files.append(item)

    # Filter out merged files and keep only trace files
    all_files = [f for f in all_files
                 if (f.endswith('.trace.json.gz') or f.endswith('.trace.json'))
                 and os.path.basename(f) not in ['merged.trace.json.gz', 'merged.trace.json']]

    # Remove duplicates
    all_files = list(dict.fromkeys(all_files))

    # Prefer .gz over .json for same base name
    file_dict = {}
    for f in all_files:
        base = f[:-len('.trace.json.gz')] if f.endswith('.trace.json.gz') else f[:-len('.trace.json')]
        if base not in file_dict or f.endswith('.gz'):
            file_dict[base] = f

    return list(file_dict.values())


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple trace.json/trace.json.gz files into a single timeline-aligned trace.',
        epilog='''
Examples:
  merge_xplane_traces.py ./traces
  merge_xplane_traces.py ./node*.trace.json.gz
  merge_xplane_traces.py ./traces1 ./traces2 -o output.trace.json.gz
  merge_xplane_traces.py node0.trace.json.gz node1.trace.json.gz --output combined.trace.json.gz

Paths can be directories, specific files, or wildcards.
Supports both .trace.json and .trace.json.gz files.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('paths', nargs='+', help='Input paths (directories, files, or wildcards)')
    parser.add_argument('-o', '--output', default='merged.trace.json.gz',
                        help='Output file path (default: merged.trace.json.gz in current directory)')

    args = parser.parse_args()

    # Resolve output path
    output_path = args.output if os.path.isabs(args.output) else os.path.join(os.getcwd(), args.output)

    # Ensure output_path has .trace.json.gz suffix
    if not output_path.endswith('.trace.json.gz'):
        if output_path.endswith('.trace'):
            output_path += '.json.gz'
        elif output_path.endswith('.trace.json'):
            output_path += '.gz'
        elif output_path.endswith('.json.gz'):
            # Insert .trace before .json.gz
            output_path = output_path[:-8] + '.trace.json.gz'
        elif output_path.endswith('.gz'):
            # Insert .trace.json before .gz
            output_path = output_path[:-3] + '.trace.json.gz'
        elif output_path.endswith('.json'):
            # Replace .json with .trace.json.gz
            output_path = output_path[:-5] + '.trace.json.gz'
        else:
            # No recognized suffix, append the full suffix
            output_path += '.trace.json.gz'

    # Collect all trace files
    all_files = collect_trace_files(args.paths)

    if not all_files:
        print("Error: No .trace.json or .trace.json.gz files found")
        sys.exit(1)

    print(f"Found {len(all_files)} trace files:")
    for f in all_files:
        print(f"  - {f}")
    print(f"\nOutput will be written to: {output_path}\n")

    merge_xplane_traces(all_files, output_path)


if __name__ == '__main__':
    main()
