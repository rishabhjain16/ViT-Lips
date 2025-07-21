#!/usr/bin/env python3
"""
Simple test script for LRS3 dataset - no complex imports needed
"""

import os

def test_data_structure(data_dir):
    """Test the basic data structure"""
    print("🔍 Testing LRS3 Data Structure")
    print("=" * 40)
    print(f"Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("❌ Directory does not exist!")
        return False
    
    # Check all splits and file types
    splits = ["train", "valid", "test"]
    file_types = ["tsv", "phn", "wrd"]  # wrd for word/sentence labels
    
    print("\n📄 Checking files for all splits:")
    all_files_status = {}
    
    for split in splits:
        print(f"\n  📁 {split.upper()} split:")
        split_status = {}
        for file_type in file_types:
            file_name = f"{split}.{file_type}"
            file_path = os.path.join(data_dir, file_name)
            exists = os.path.exists(file_path)
            status = "✓" if exists else "❌"
            print(f"    {status} {file_name}")
            split_status[file_type] = exists
        all_files_status[split] = split_status
    
    # Check dictionary files
    print(f"\n  📚 Dictionary files:")
    dict_files = ["dict.phn.txt", "dict.wrd.txt"]
    for dict_file in dict_files:
        file_path = os.path.join(data_dir, dict_file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "❌"
        print(f"    {status} {dict_file}")
    
    return all_files_status
    
def test_manifest_file(data_dir, split="valid"):
    """Test reading manifest file"""
    print(f"\n📊 Testing manifest file ({split}.tsv):")
    try:
        manifest_path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(manifest_path):
            print(f"  ⚠️ {split}.tsv not found, skipping...")
            return False
            
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print("❌ Manifest file is too short")
            return False
        
        root_path = lines[0].strip()
        print(f"  Root path: {root_path}")
        print(f"  Number of samples: {len(lines) - 1}")
        
        # Check first few entries
        print("  First 3 samples:")
        for i in range(1, min(4, len(lines))):
            parts = lines[i].strip().split('\t')
            if len(parts) >= 5:
                file_id = parts[0]
                video_path = parts[1]
                audio_path = parts[2]
                frame_count = parts[3]
                audio_samples = parts[4]
                print(f"    {i}: {file_id}")
                print(f"        Video: {video_path}")
                print(f"        Audio: {audio_path}")
                print(f"        Frames: {frame_count}")
            else:
                print(f"    {i}: Invalid format (expected 5 columns, got {len(parts)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading manifest: {e}")
        return False


def test_label_files(data_dir, split="valid"):
    """Test reading label files (both phonemes and words)"""
    
    # Test phoneme labels
    print(f"\n📝 Testing phoneme labels ({split}.phn):")
    try:
        phn_path = os.path.join(data_dir, f"{split}.phn")
        if not os.path.exists(phn_path):
            print(f"  ⚠️ {split}.phn not found, skipping...")
        else:
            with open(phn_path, 'r') as f:
                phn_labels = f.readlines()
            
            print(f"  Number of phoneme labels: {len(phn_labels)}")
            print("  First 3 phoneme sequences:")
            for i in range(min(3, len(phn_labels))):
                label = phn_labels[i].strip()
                phonemes = label.split()
                print(f"    {i+1}: {len(phonemes)} phonemes - {' '.join(phonemes[:10])}...")
                
    except Exception as e:
        print(f"❌ Error reading phoneme labels: {e}")
    
    # Test word/sentence labels
    print(f"\n📄 Testing word/sentence labels ({split}.wrd):")
    try:
        wrd_path = os.path.join(data_dir, f"{split}.wrd")
        if not os.path.exists(wrd_path):
            print(f"  ⚠️ {split}.wrd not found, skipping...")
        else:
            with open(wrd_path, 'r') as f:
                wrd_labels = f.readlines()
            
            print(f"  Number of word labels: {len(wrd_labels)}")
            print("  First 3 sentences:")
            for i in range(min(3, len(wrd_labels))):
                sentence = wrd_labels[i].strip()
                words = sentence.split()
                print(f"    {i+1}: {len(words)} words - \"{sentence[:80]}{'...' if len(sentence) > 80 else ''}\"")
                
    except Exception as e:
        print(f"❌ Error reading word labels: {e}")


def test_dictionaries(data_dir):
    """Test phoneme and word dictionaries"""
    
    # Test phoneme dictionary
    print(f"\n🔤 Testing phoneme dictionary (dict.phn.txt):")
    try:
        dict_path = os.path.join(data_dir, "dict.phn.txt")
        if not os.path.exists(dict_path):
            print("  ⚠️ dict.phn.txt not found, skipping...")
        else:
            with open(dict_path, 'r') as f:
                dict_lines = f.readlines()
            
            print(f"  Number of phonemes: {len(dict_lines)}")
            print("  First 10 phonemes:")
            for i in range(min(10, len(dict_lines))):
                line = dict_lines[i].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        phoneme, idx = parts[0], parts[1]
                        print(f"    {phoneme} -> {idx}")
        
    except Exception as e:
        print(f"❌ Error reading phoneme dictionary: {e}")
    
    # Test word dictionary
    print(f"\n📚 Testing word dictionary (dict.wrd.txt):")
    try:
        dict_path = os.path.join(data_dir, "dict.wrd.txt")
        if not os.path.exists(dict_path):
            print("  ⚠️ dict.wrd.txt not found, skipping...")
        else:
            with open(dict_path, 'r') as f:
                dict_lines = f.readlines()
            
            print(f"  Number of words: {len(dict_lines)}")
            print("  First 10 words:")
            for i in range(min(10, len(dict_lines))):
                line = dict_lines[i].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        word, idx = parts[0], parts[1]
                        print(f"    {word} -> {idx}")
        
    except Exception as e:
        print(f"❌ Error reading word dictionary: {e}")

def test_media_files(data_dir, split="valid"):
    """Test if video and audio files exist"""
    print(f"\n🎬 Testing media files for {split} split:")
    
    try:
        manifest_path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(manifest_path):
            print(f"  ⚠️ {split}.tsv not found, skipping...")
            return
            
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        
        root_path = lines[0].strip()
        print(f"  Root path: {root_path}")
        
        # Check first few video and audio files
        check_count = min(5, len(lines) - 1)
        video_existing = 0
        audio_existing = 0
        
        print(f"  Checking first {check_count} media files:")
        for i in range(1, check_count + 1):
            parts = lines[i].strip().split('\t')
            if len(parts) >= 5:
                file_id = parts[0]
                video_path = parts[1]  # Already absolute path
                audio_path = parts[2]  # Already absolute path
                
                video_exists = os.path.exists(video_path)
                audio_exists = os.path.exists(audio_path)
                
                v_status = "✓" if video_exists else "❌"
                a_status = "✓" if audio_exists else "❌"
                
                print(f"    {file_id}:")
                print(f"      {v_status} Video: {os.path.basename(video_path)}")
                print(f"      {a_status} Audio: {os.path.basename(audio_path)}")
                
                if video_exists:
                    video_existing += 1
                if audio_exists:
                    audio_existing += 1
        
        print(f"  Summary for {split}:")
        print(f"    Videos found: {video_existing}/{check_count}")
        print(f"    Audio found: {audio_existing}/{check_count}")
        
    except Exception as e:
        print(f"❌ Error checking media files: {e}")


def get_split_statistics(data_dir):
    """Get statistics for all splits"""
    print(f"\n📈 Dataset Statistics Summary:")
    print("=" * 40)
    
    splits = ["train", "valid", "test"]
    total_samples = 0
    
    for split in splits:
        manifest_path = os.path.join(data_dir, f"{split}.tsv")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                lines = f.readlines()
            
            num_samples = len(lines) - 1  # Subtract header line
            total_samples += num_samples
            
            # Get frame statistics
            frame_counts = []
            for i in range(1, len(lines)):
                parts = lines[i].strip().split('\t')
                if len(parts) >= 4:
                    frame_counts.append(int(parts[3]))
            
            if frame_counts:
                min_frames = min(frame_counts)
                max_frames = max(frame_counts)
                avg_frames = sum(frame_counts) / len(frame_counts)
                
                print(f"  📁 {split.upper()}: {num_samples:,} samples")
                print(f"      Frames: {min_frames}-{max_frames} (avg: {avg_frames:.1f})")
            else:
                print(f"  📁 {split.upper()}: {num_samples:,} samples")
        else:
            print(f"  📁 {split.upper()}: Not found")
    
    print(f"\n  🎯 Total samples: {total_samples:,}")


def main():
    data_dir = "/home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face"
    
    print("🚀 Comprehensive LRS3 Dataset Test")
    print("=" * 60)
    
    # Test data structure for all splits
    all_files_status = test_data_structure(data_dir)
    
    # Test dictionaries
    test_dictionaries(data_dir)
    
    # Get overall statistics
    get_split_statistics(data_dir)
    
    # Test each available split
    splits = ["train", "valid", "test"]
    for split in splits:
        split_files = all_files_status.get(split, {})
        if split_files.get('tsv', False):  # Check if TSV file exists for this split
            print(f"\n{'='*20} {split.upper()} SPLIT {'='*20}")
            test_manifest_file(data_dir, split)
            test_label_files(data_dir, split)
            test_media_files(data_dir, split)
    
    print(f"\n{'='*60}")
    print("✨ Comprehensive test complete!")
    print("💡 Now you can test the dataset classes with:")
    print("   python datasets/test_dataset.py /home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
