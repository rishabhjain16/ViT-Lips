#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import LipReadingDataset

def test_split(data_dir, split):
    print("\n=== " + split.upper() + " ===")
    
    # PHONEME mode
    print("PHONEME:")
    dataset_phn = LipReadingDataset(data_dir=data_dir, split=split, label_type="phn", load_audio=True, use_logfbank=False)
    
    if len(dataset_phn) > 0:
        sample = dataset_phn[0]
        raw_label = dataset_phn.labels[0]
        print("  Raw phonemes: " + raw_label)
        print("  Converted: " + str(sample["label"]))
        print("  Video: " + str(sample["video"].shape))
        if sample["audio"] is not None:
            print("  Audio (raw): " + str(sample["audio"].shape))
        else:
            print("  Audio: None")
    
    # Test LOGFBANK features
    print("PHONEME (logfbank):")
    try:
        dataset_phn_logfbank = LipReadingDataset(data_dir=data_dir, split=split, label_type="phn", load_audio=True, use_logfbank=True)
        if len(dataset_phn_logfbank) > 0:
            sample = dataset_phn_logfbank[0]
            if sample["audio"] is not None:
                print("  Audio (logfbank): " + str(sample["audio"].shape))
            else:
                print("  Audio (logfbank): None")
    except Exception as e:
        print("  Audio (logfbank): Error - " + str(e)[:50] + "...")
    
    # SENTENCE mode  
    print("SENTENCE:")
    dataset_wrd = LipReadingDataset(data_dir=data_dir, split=split, label_type="wrd", load_audio=True, use_logfbank=False)
    
    if len(dataset_wrd) > 0:
        sample = dataset_wrd[0]
        raw_label = dataset_wrd.labels[0]
        print("  Raw sentence: " + raw_label)
        print("  Converted: " + sample["label"])
        if sample["audio"] is not None:
            print("  Audio: " + str(sample["audio"].shape))
        else:
            print("  Audio: None")

def main():
    data_dir = "/home/rishabh/Desktop/Datasets/lrs3/433h_data_full_face"
    for split in ["train", "valid", "test"]:
        if os.path.exists(data_dir + "/" + split + ".tsv"):
            test_split(data_dir, split)

if __name__ == "__main__":
    main()
