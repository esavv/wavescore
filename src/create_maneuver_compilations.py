import os
import cv2
import pandas as pd
import glob
import argparse
from pathlib import Path
import time
import re

def create_maneuver_compilations(base_data_dir, output_dir, verbose=False):
    """
    Creates compilation videos for each maneuver type by stitching together
    all sequences with the same label in their original order.
    
    Args:
        base_data_dir: Directory containing heat and ride data with sequences
        output_dir: Directory to save the compilation videos
        verbose: Whether to print detailed debug information
    """
    start_time = time.time()
    print(f"Starting compilation creation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize collections to store sequences by maneuver type
    sequences_by_maneuver = {}
    
    # Load the maneuver taxonomy from CSV
    taxonomy_path = os.path.join(base_data_dir, "maneuver_taxonomy.csv")
    maneuver_names = {}
    
    if os.path.exists(taxonomy_path):
        try:
            taxonomy_df = pd.read_csv(taxonomy_path)
            for _, row in taxonomy_df.iterrows():
                # Convert spaces to underscores and make lowercase for filenames
                name = row['name'].lower().replace(' ', '_')
                maneuver_names[row['id']] = name
            print(f"Loaded maneuver taxonomy with {len(maneuver_names)} types")
            if verbose:
                print(f"Maneuver names: {maneuver_names}")
        except Exception as e:
            print(f"Error loading taxonomy: {e}")
            print("Please ensure the taxonomy file exists and has 'id' and 'name' columns.")
            return
    else:
        print(f"Taxonomy file not found at {taxonomy_path}")
        print("Please ensure maneuver_taxonomy.csv exists in the data directory.")
        return
    
    # Find all CSV files with sequence labels
    label_files = glob.glob(f"{base_data_dir}/**/*_seq_labels.csv", recursive=True)
    print(f"Found {len(label_files)} sequence label files")
    
    if len(label_files) == 0:
        print(f"No label files found in {base_data_dir}")
        return
    
    # Walk through the directory structure to find all sequence directories and their labels
    all_sequences = []
    heat_count = 0
    ride_count = 0
    current_heat = None
    
    # Process each label file
    for label_file in label_files:
        # Extract heat and ride information from the path
        path_parts = Path(label_file).parts
        
        # Find the indices of 'heats' and 'rides' in the path
        try:
            heat_idx = path_parts.index('heats')
            ride_idx = path_parts.index('rides')
            
            heat_id = path_parts[heat_idx + 1]
            ride_id = path_parts[ride_idx + 1]
            
            # Update heat/ride counting for progress display
            if current_heat != heat_id:
                current_heat = heat_id
                heat_count += 1
                if verbose:
                    print(f"Processing heat: {heat_id}")
                else:
                    print(f"Processing heat {heat_count}...", end="\r")
            
            ride_count += 1
        except (ValueError, IndexError):
            if verbose:
                print(f"Warning: Could not extract heat/ride info from {label_file}")
            continue
        
        # Load the label file
        try:
            labels_df = pd.read_csv(label_file)
            
            # Rename columns if they don't match expected names
            if 'sequence_id' not in labels_df.columns and 'sequence' in labels_df.columns:
                labels_df = labels_df.rename(columns={'sequence': 'sequence_id'})
            
            if 'label' not in labels_df.columns and 'maneuver_id' in labels_df.columns:
                labels_df = labels_df.rename(columns={'maneuver_id': 'label'})
                
        except Exception as e:
            if verbose:
                print(f"Error reading CSV file {label_file}: {e}")
            continue
        
        # Get the base path for sequences
        ride_dir = os.path.dirname(label_file)
        seqs_dir = os.path.join(ride_dir, "seqs")  # Update to look in the 'seqs' subdirectory
        
        # Process each sequence
        for _, row in labels_df.iterrows():
            # Check if we have the expected columns
            if 'sequence_id' not in row or 'label' not in row:
                continue
                
            sequence_id = row['sequence_id']
            label = row['label']
            
            # Skip if this label is not in our taxonomy
            if label not in maneuver_names:
                continue
                
            # Build the full path to the sequence directory using the seqs subdirectory
            seq_path = os.path.join(seqs_dir, sequence_id)
            
            # Skip if the directory doesn't exist
            if not os.path.exists(seq_path):
                # Try alternative formats
                alternatives = [
                    os.path.join(ride_dir, sequence_id),  # Directly in ride dir 
                    os.path.join(ride_dir, "seqs", f"seq_{sequence_id}"),  # With seq_ prefix in seqs dir
                    os.path.join(ride_dir, f"seq_{sequence_id}")  # With seq_ prefix in ride dir
                ]
                
                found = False
                for alt_path in alternatives:
                    if os.path.exists(alt_path):
                        seq_path = alt_path
                        found = True
                        break
                        
                if not found:
                    continue
            
            # Extract sequence number for proper numeric sorting
            seq_num = 0
            if 'seq_' in sequence_id:
                # Extract the number from sequence_id (e.g., "seq_10" -> 10)
                seq_match = re.search(r'seq_(\d+)', sequence_id)
                if seq_match:
                    seq_num = int(seq_match.group(1))
            else:
                # If sequence_id is just a number
                try:
                    seq_num = int(sequence_id)
                except ValueError:
                    # Keep as 0 if not a number
                    pass
                
            # Add to our collection with ordering info
            all_sequences.append({
                'heat_id': heat_id,
                'ride_id': ride_id,
                'sequence_id': sequence_id,
                'seq_num': seq_num,  # Add numeric sequence number for proper sorting
                'label': label,
                'path': seq_path
            })
    
    print(f"\nProcessed {heat_count} heats, {ride_count} rides, found {len(all_sequences)} valid sequences")
    
    if len(all_sequences) == 0:
        print("No sequences found to process. Check your data directory structure.")
        return
    
    # Sort all sequences properly with numeric sorting for sequence_id
    all_sequences.sort(key=lambda x: (x['heat_id'], x['ride_id'], x['seq_num']))
    
    # Group sequences by maneuver type while preserving order
    sequences_by_maneuver = {}
    for seq in all_sequences:
        label = seq['label']
        if label not in sequences_by_maneuver:
            sequences_by_maneuver[label] = []
        sequences_by_maneuver[label].append(seq)
    
    if verbose:
        print(f"Maneuver types found: {list(sequences_by_maneuver.keys())}")
    
    # Create a video for each maneuver type
    maneuver_count = len(sequences_by_maneuver)
    for i, (label, sequences) in enumerate(sequences_by_maneuver.items(), 1):
        maneuver_name = maneuver_names.get(label, f"unknown_{label}")
        output_file = os.path.join(output_dir, f"{maneuver_name}.mp4")
        
        # Sort sequences by heat, ride, and sequence number to ensure proper ordering
        sequences.sort(key=lambda x: (x['heat_id'], x['ride_id'], x['seq_num']))
        sequence_paths = [seq['path'] for seq in sequences]
        
        # Add debugging for sequence ordering
        if verbose:
            print("\nSequence order for", maneuver_name)
            for seq in sequences:
                print(f"  Heat: {seq['heat_id']}, Ride: {seq['ride_id']}, Seq: {seq['sequence_id']} (Num: {seq['seq_num']})")
        
        print(f"Creating video [{i}/{maneuver_count}]: {maneuver_name}.mp4 ({len(sequence_paths)} sequences)")
        
        # Skip if no sequences
        if not sequence_paths:
            continue
            
        # Initialize video writer
        frame_height, frame_width = None, None
        video_writer = None
        frame_count = 0
        
        # Process each sequence
        for seq_idx, seq_dir in enumerate(sequence_paths):
            if verbose:
                print(f"  Processing sequence {seq_idx+1}/{len(sequence_paths)}: {os.path.basename(seq_dir)}")
            elif seq_idx % 10 == 0 or seq_idx == len(sequence_paths) - 1:
                # Print progress every 10 sequences or for the last one
                print(f"  Progress: {seq_idx+1}/{len(sequence_paths)} sequences", end="\r")
            
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
            
            if not frame_files:
                continue
                
            # Process frames in this sequence
            for frame_path in frame_files:
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                    
                # Initialize video writer with first frame dimensions
                if video_writer is None:
                    frame_height, frame_width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    try:
                        video_writer = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))
                    except Exception as e:
                        print(f"  Error creating video writer: {e}")
                        break
                
                # Resize if dimensions don't match (unlikely but just in case)
                if frame.shape[:2] != (frame_height, frame_width):
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Write the frame
                try:
                    video_writer.write(frame)
                    frame_count += 1
                except Exception as e:
                    if verbose:
                        print(f"  Error writing frame: {e}")
        
        # Release the video writer
        if video_writer is not None:
            video_writer.release()
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
            print(f"  Saved {output_file} ({frame_count} frames, {file_size:.2f} MB)")
        else:
            print(f"  No frames processed for {maneuver_name}")
    
    elapsed_time = time.time() - start_time
    print(f"Compilation complete! {len(sequences_by_maneuver)} videos created in {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    # Set up paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Create compilation videos by maneuver type")
    parser.add_argument("--data_dir", type=str, help="Base directory containing heat and ride data")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data/sequence_vids"), 
                        help="Directory to save compilation videos")
    parser.add_argument("--taxonomy_file", type=str, help="Path to maneuver taxonomy CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # If data_dir not provided, use default relative to project root
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, "data")
    
    if args.verbose:
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
    
    create_maneuver_compilations(args.data_dir, args.output_dir, args.verbose)
    print(f"Videos saved to: {args.output_dir}") 