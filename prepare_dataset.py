"""
prepare_dataset.py
------------------
Prepares the MIT-BIH Arrhythmia Database for ECG beat classification:
1. Downloads the 44 MIT-BIH records (excluding 4 paced records) using wfdb.
2. Extracts MLII lead ECG signal for each record.
3. Filters annotations to valid beats and maps them to AAMI EC57 classes:
     N: Normal, LBBB, RBBB, atrial escape, nodal escape (N, L, R, e, j)
     S: Atrial premature, aberrated atrial, nodal premature, supraventricular premature (A, a, J, S)
     V: Premature ventricular contraction, ventricular escape (V, E)
     F: Fusion of ventricular and normal (F)
     Q: Paced, fusion of paced, unclassifiable (/, f, Q)
   Non-beat annotations (e.g. rhythm changes, noise) are ignored.
4. Slices a 200-sample window around each R-peak (90 samples before, 110 after at 360 Hz)
   and applies z-score normalization.
5. Computes RR timing features: previous RR, next RR, average of the last 10 RRs,
   and the ratio of previous RR to the average.
6. Splits data strictly by patient into:
     - DS1 Train (18 records)
     - DS1 Validation (4 records held out from DS1)
     - DS2 Test (22 records, untouched benchmark)
7. Computes and saves feature scalers fit on DS1 Train only, then saves processed arrays.
"""

import os
import json
import random
import argparse
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import joblib
import wfdb

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------

# Paced records to exclude completely per standard AAMI EC57 protocol
PACED_RECORDS = ['102', '104', '107', '217']

# Inter-patient division according to de Chazal et al. (2004)
# DS1: 22 records used for training and development
DS1_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]

# DS2: 22 records reserved strictly for final testing (never tuned on)
DS2_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# Hold out a representative subset of DS1 patients for validation:
# Record 115 (normal baseline), 118 (routine + S beats), 201 (mixed N, S, V), 205 (mixed N, V, F)
VAL_RECORDS = ['115', '118', '201', '205']
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]

# AAMI EC57 class mapping
# 0 = N (Non-ectopic / Normal)
# 1 = S (Supraventricular ectopic)
# 2 = V (Ventricular ectopic)
# 3 = F (Fusion of ventricular and normal)
# 4 = Q (Unknown / Paced / Unclassifiable)
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
SAMPLING_RATE = 360  # MIT-BIH sampling frequency in Hz
SAMPLES_BEFORE = 90  # 90 samples before R-peak (250 ms)
SAMPLES_AFTER = 110  # 110 samples after R-peak (305 ms)
WINDOW_SIZE = SAMPLES_BEFORE + SAMPLES_AFTER  # 200 samples total


# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------

def download_mitdb_if_needed(raw_data_dir: str):
    """
    Downloads all required 44 MIT-BIH records using wfdb if not already present.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    all_needed_records = sorted(list(set(DS1_RECORDS + DS2_RECORDS)))
    
    missing = []
    for rec_id in all_needed_records:
        header_file = os.path.join(raw_data_dir, f"{rec_id}.hea")
        data_file = os.path.join(raw_data_dir, f"{rec_id}.dat")
        ann_file = os.path.join(raw_data_dir, f"{rec_id}.atr")
        if not (os.path.exists(header_file) and os.path.exists(data_file) and os.path.exists(ann_file)):
            missing.append(rec_id)
            
    if missing:
        print(f"Downloading {len(missing)} missing records into '{raw_data_dir}' from PhysioNet...")
        wfdb.dl_database('mitdb', raw_data_dir, records=missing)
        print("Download complete.")
    else:
        print(f"All 44 MIT-BIH records are already available in '{raw_data_dir}'.")


def extract_features_from_record(record_id: str, raw_data_dir: str):
    """
    Reads a single MIT-BIH record:
    - Finds the MLII lead.
    - Filters annotations to valid AAMI beats.
    - Slices 200-sample windows around R-peaks with z-score normalization.
    - Computes RR features (prev, next, 10-beat average, ratio).
    - Extracts simple summary features for baseline models.
    """
    record_path = os.path.join(raw_data_dir, record_id)
    
    # 1. Read the ECG record and locate the MLII channel
    rec = wfdb.rdrecord(record_path)
    sig_names_upper = [s.upper() for s in rec.sig_name]
    
    if 'MLII' in sig_names_upper:
        lead_idx = sig_names_upper.index('MLII')
    else:
        raise ValueError(f"Record {record_id} does not contain lead MLII! Found leads: {rec.sig_name}")
        
    signal = rec.p_signal[:, lead_idx]
    sig_len = len(signal)
    
    # 2. Read annotations and keep only valid AAMI beat annotations
    ann = wfdb.rdann(record_path, 'atr')
    valid_beats = []  # stores (sample_idx, aami_label, symbol)
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in AAMI_MAPPING:
            valid_beats.append((int(sample), AAMI_MAPPING[symbol], symbol))
            
    num_beats = len(valid_beats)
    if num_beats == 0:
        return [], [], [], []
        
    sample_indices = np.array([b[0] for b in valid_beats])
    
    # 3. Precompute RR intervals between consecutive valid beats (in seconds)
    rr_intervals = []
    for i in range(num_beats - 1):
        rr_intervals.append((sample_indices[i + 1] - sample_indices[i]) / float(SAMPLING_RATE))
        
    # Default interval in seconds (0.8s ~ 75 bpm) if record has single beat
    default_rr = 0.8
    
    waveforms = []
    rr_features = []
    simple_features = []
    labels = []
    
    for i in range(num_beats):
        r_peak = sample_indices[i]
        label = valid_beats[i][1]
        
        # Check boundary condition for window extraction
        start_idx = r_peak - SAMPLES_BEFORE
        end_idx = r_peak + SAMPLES_AFTER
        if start_idx < 0 or end_idx > sig_len:
            # Skip beats too close to the very start or end of the 30-min recording
            continue
            
        # Cut 200-sample waveform around R-peak
        beat_raw = signal[start_idx:end_idx]
        
        # Z-score normalize the beat waveform
        beat_mean = np.mean(beat_raw)
        beat_std = np.std(beat_raw)
        beat_norm = (beat_raw - beat_mean) / (beat_std + 1e-8)
        
        # Compute RR timing features:
        # a) Previous RR interval
        if i > 0:
            rr_prev = (sample_indices[i] - sample_indices[i - 1]) / float(SAMPLING_RATE)
        elif len(rr_intervals) > 0:
            rr_prev = rr_intervals[0]
        else:
            rr_prev = default_rr
            
        # b) Next RR interval
        if i < num_beats - 1:
            rr_next = (sample_indices[i + 1] - sample_indices[i]) / float(SAMPLING_RATE)
        elif len(rr_intervals) > 0:
            rr_next = rr_intervals[-1]
        else:
            rr_next = default_rr
            
        # c) Average of the last 10 RR intervals
        if i == 0:
            rr_avg_10 = rr_prev
        else:
            start_j = max(0, i - 10)
            history = [(sample_indices[j + 1] - sample_indices[j]) / float(SAMPLING_RATE)
                       for j in range(start_j, i)]
            rr_avg_10 = float(np.mean(history)) if len(history) > 0 else rr_prev
            
        # d) Ratio of previous RR to the 10-beat average
        rr_ratio = rr_prev / (rr_avg_10 + 1e-8)
        
        rr_feat = [rr_prev, rr_next, rr_avg_10, rr_ratio]
        
        # Compute handcrafted simple features for the baseline model:
        # Statistical moments, energy, peak-to-peak, zero crossings, and downsampled shape
        ptp_val = float(np.ptp(beat_norm))
        min_val = float(np.min(beat_norm))
        max_val = float(np.max(beat_norm))
        energy = float(np.sum(beat_norm ** 2))
        qrs_energy = float(np.sum(beat_norm[80:120] ** 2))  # Energy around R-peak complex
        skew_val = float(stats.skew(beat_norm))
        kurt_val = float(stats.kurtosis(beat_norm))
        zero_crossings = float(np.sum(np.diff(np.sign(beat_norm)) != 0))
        # 20 evenly spaced downsampled points along the beat
        downsampled = beat_norm[::10].tolist()  # 200 // 10 = 20 points
        
        stat_feats = [ptp_val, min_val, max_val, energy, qrs_energy, skew_val, kurt_val, zero_crossings]
        combined_simple = rr_feat + stat_feats + downsampled  # 4 + 8 + 20 = 32 features
        
        waveforms.append(beat_norm.astype(np.float32))
        rr_features.append(rr_feat)
        simple_features.append(combined_simple)
        labels.append(label)
        
    return waveforms, rr_features, simple_features, labels


def process_dataset(raw_data_dir: str, output_dir: str):
    """
    Builds and saves the train, validation, and test datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- 1. Downloading/Verifying MIT-BIH Database ---")
    download_mitdb_if_needed(raw_data_dir)
    
    print("\n--- 2. Extracting Beats and Features ---")
    splits = {
        'train': TRAIN_RECORDS,
        'val': VAL_RECORDS,
        'test': DS2_RECORDS
    }
    
    extracted_data = {}
    
    for split_name, record_list in splits.items():
        print(f"\nProcessing {split_name.upper()} split ({len(record_list)} records: {record_list})...")
        all_waves = []
        all_rr = []
        all_simple = []
        all_labels = []
        
        for r_id in record_list:
            if r_id in PACED_RECORDS:
                print(f"Skipping paced record {r_id}")
                continue
                
            waves, rrs, simples, lbls = extract_features_from_record(r_id, raw_data_dir)
            all_waves.extend(waves)
            all_rr.extend(rrs)
            all_simple.extend(simples)
            all_labels.extend(lbls)
            
        all_waves = np.array(all_waves, dtype=np.float32)
        all_rr = np.array(all_rr, dtype=np.float32)
        all_simple = np.array(all_simple, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        extracted_data[split_name] = {
            'waves': all_waves,
            'rr': all_rr,
            'simple': all_simple,
            'labels': all_labels
        }
        
        # Display class distribution
        counts = {CLASS_NAMES[i]: int(np.sum(all_labels == i)) for i in range(len(CLASS_NAMES))}
        print(f"{split_name.upper()} Total beats: {len(all_labels)} | Class distribution: {counts}")

    print("\n--- 3. Normalizing RR & Simple Features using Train Scaler ---")
    # Standardize RR features and Simple features using only DS1 Train
    rr_scaler = StandardScaler()
    simple_scaler = StandardScaler()
    
    rr_scaler.fit(extracted_data['train']['rr'])
    simple_scaler.fit(extracted_data['train']['simple'])
    
    # Save the fitted scalers
    joblib.dump(rr_scaler, os.path.join(output_dir, 'rr_scaler.joblib'))
    joblib.dump(simple_scaler, os.path.join(output_dir, 'simple_scaler.joblib'))
    print("Fitted StandardScaler on DS1 Train and saved scalers.")
    
    # Transform each split
    for split_name in splits:
        extracted_data[split_name]['rr_scaled'] = rr_scaler.transform(extracted_data[split_name]['rr']).astype(np.float32)
        extracted_data[split_name]['simple_scaled'] = simple_scaler.transform(extracted_data[split_name]['simple']).astype(np.float32)
        
        # Save as npz file
        save_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            save_path,
            waveform=extracted_data[split_name]['waves'],
            rr=extracted_data[split_name]['rr_scaled'],
            simple=extracted_data[split_name]['simple_scaled'],
            labels=extracted_data[split_name]['labels']
        )
        print(f"Saved {split_name} split to {save_path} (Waveform shape: {extracted_data[split_name]['waves'].shape})")
        
    # Save dataset metadata summary
    metadata = {
        'sampling_rate': SAMPLING_RATE,
        'window_size': WINDOW_SIZE,
        'samples_before': SAMPLES_BEFORE,
        'samples_after': SAMPLES_AFTER,
        'classes': CLASS_NAMES,
        'splits': {
            'train_records': TRAIN_RECORDS,
            'val_records': VAL_RECORDS,
            'test_records': DS2_RECORDS,
            'excluded_paced_records': PACED_RECORDS
        },
        'counts': {
            split: {
                'total': int(len(extracted_data[split]['labels'])),
                'class_counts': {CLASS_NAMES[i]: int(np.sum(extracted_data[split]['labels'] == i)) for i in range(len(CLASS_NAMES))}
            }
            for split in splits
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata summary to {metadata_path}")
    print("\nDataset preparation completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH dataset for ECG beat classification.")
    parser.add_argument('--raw_dir', type=str, default='data/mitdb', help="Directory containing raw MIT-BIH files")
    parser.add_argument('--out_dir', type=str, default='data/processed', help="Output directory for processed files")
    args = parser.parse_args()
    
    process_dataset(args.raw_dir, args.out_dir)

