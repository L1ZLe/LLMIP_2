#!/usr/bin/env python3
"""
IEEE 118-Bus Power Flow Data Preparation for LLMIP Pipeline
===========================================================
Extracts real power flow features and voltage targets from the
https://github.com/evgenytsydenov/ieee118_power_flow_data dataset.

What this script produces:
- Features: load by region (R1/R2/R3), generation by type and region,
            slack bus state, temporal features, per-bus voltage profile stats
- Targets: vm_pu (voltage magnitude) for all 118 buses
- 200 hourly snapshots sampled across 2024

This replaces the synthetic data in data/prepared/ with REAL power flow data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = Path(__file__).parent.parent / "data" / "grid"
SAMPLES_DIR = RAW_DIR / "samples"
PREPARED_DIR = PROJECT_DIR / "data" / "prepared"
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 200
RANDOM_STATE = 42

GENERATOR_TYPES = {
    'solar', 'wind', 'hydro', 'biomass',
    'combined', 'combustion', 'internal', 'steam',
    'geothermal', 'oil'
}


def parse_pandapower_json(json_path):
    """Load a pandapowerNet JSON file and extract key data frames.

    The JSON uses a nested pandas split-format serialization.
    Each DataFrame is stored at obj['_object'] as a JSON string.
    """
    with open(json_path) as f:
        raw = json.load(f)

    def parse_df(key):
        obj = raw['_object'].get(key, {})
        if not obj or '_object' not in obj:
            return None
        s = obj['_object']
        if not isinstance(s, str):
            return None
        try:
            d = json.loads(s)
            return pd.DataFrame(d['data'], columns=d['columns'], index=d['index'])
        except (json.JSONDecodeError, KeyError):
            return None

    res_bus = parse_df('res_bus')
    res_load = parse_df('res_load')
    res_gen = parse_df('res_gen')
    load_bus = parse_df('load')
    gen_bus = parse_df('gen')
    ext_grid = parse_df('ext_grid')

    converged = raw['_object'].get('converged', False)

    return {
        'res_bus': res_bus,
        'res_load': res_load,
        'res_gen': res_gen,
        'load_bus': load_bus,
        'gen_bus': gen_bus,
        'ext_grid': ext_grid,
        'converged': converged,
    }


def extract_bus_metadata():
    """Load static bus metadata (region, voltage level)."""
    buses = pd.read_csv(RAW_DIR / 'buses.csv')

    bus_meta = {}
    for _, row in buses.iterrows():
        idx = int(row['bus_name'].replace('bus_', '')) - 1
        bus_meta[idx] = {
            'region': row['region'],
            'v_rated_kv': row['v_rated_kv'],
            'is_slack': row['is_slack'],
        }
    return bus_meta


def extract_gen_metadata():
    """Load static generator metadata (type, bus)."""
    gens = pd.read_csv(RAW_DIR / 'gens.csv')
    gen_meta = {}
    for _, row in gens.iterrows():
        gen_type = row['gen_name'].split('_')[0].lower()
        if gen_type not in GENERATOR_TYPES:
            gen_type = 'other'
        bus_idx = int(row['bus_name'].replace('bus_', '')) - 1
        gen_meta[row['gen_name']] = {
            'type': gen_type,
            'bus_idx': bus_idx,
        }
    return gen_meta


def load_load_metadata():
    """Load static load metadata (bus mapping)."""
    loads = pd.read_csv(RAW_DIR / 'loads.csv')
    load_meta = {}
    for _, row in loads.iterrows():
        bus_idx = int(row['bus_name'].replace('bus_', '')) - 1
        load_meta[row['load_name']] = bus_idx
    return load_meta


def _gen_types_for_category(category, gen_meta):
    """Map simplified generation category to actual type strings."""
    mapping = {
        'solar': {'solar'},
        'wind': {'wind'},
        'hydro': {'hydro'},
        'thermal': {
            'biomass', 'combined', 'combustion', 'internal',
            'steam', 'geothermal', 'oil'
        },
    }
    return mapping.get(category, set())


def extract_features_and_targets(sample_path, bus_meta, gen_meta, load_meta, ts):
    """Extract per-snapshot features and targets from a power flow solution.

    Args:
        sample_path: Path to the JSON file
        bus_meta: dict mapping bus index -> {region, v_rated_kv, is_slack}
        gen_meta: dict mapping gen_name -> {type, bus_idx}
        load_meta: dict mapping load_name -> bus_idx
        ts: datetime of this snapshot

    Returns:
        features: dict of scalar features
        targets: dict of {vm_bus_N: value for N in 1..118}
    """
    data = parse_pandapower_json(sample_path)

    if not data['converged']:
        return None, None

    res_bus = data['res_bus']
    res_load = data['res_load']
    res_gen = data['res_gen']
    load_df = data['load_bus']
    gen_df = data['gen_bus']

    if res_bus is None:
        return None, None

    features = {}
    targets = {}

    # ---- Temporal features ----
    features['hour'] = float(ts.hour)
    features['dayofweek'] = float(ts.weekday())
    features['month'] = float(ts.month)
    features['is_weekend'] = float(1.0 if ts.weekday() >= 5 else 0.0)
    features['hour_sin'] = float(np.sin(2 * np.pi * ts.hour / 24))
    features['hour_cos'] = float(np.cos(2 * np.pi * ts.hour / 24))
    features['month_sin'] = float(np.sin(2 * np.pi * ts.month / 12))
    features['month_cos'] = float(np.cos(2 * np.pi * ts.month / 12))

    # ---- Slack bus (ext_grid) ----
    ext = data['ext_grid']
    if ext is not None and len(ext) > 0:
        slack_row = ext.iloc[0]
        features['slack_vm_pu'] = float(slack_row.get('vm_pu', 1.2))
        features['slack_va_degree'] = float(slack_row.get('va_degree', 0.0))
        features['slack_bus'] = float(int(slack_row.get('bus', 68)))
    else:
        features['slack_vm_pu'] = 1.2
        features['slack_va_degree'] = 0.0
        features['slack_bus'] = 68.0

    # ---- Per-region load aggregates ----
    # res_load rows align 1:1 with load_df rows
    if res_load is not None and load_df is not None:
        n_loads = min(len(res_load), len(load_df))
        load_p_by_region = {'r1': 0.0, 'r2': 0.0, 'r3': 0.0}
        load_q_by_region = {'r1': 0.0, 'r2': 0.0, 'r3': 0.0}
        load_n_by_region = {'r1': 0, 'r2': 0, 'r3': 0}

        for i in range(n_loads):
            bus_idx = int(load_df['bus'].iat[i])
            p_mw = float(res_load['p_mw'].iat[i])
            q_mvar = float(res_load['q_mvar'].iat[i])
            region = bus_meta.get(bus_idx, {}).get('region', None)
            if region in load_p_by_region:
                load_p_by_region[region] += p_mw
                load_q_by_region[region] += q_mvar
                load_n_by_region[region] += 1

        for region in ['r1', 'r2', 'r3']:
            features[f'load_p_{region}_mw'] = load_p_by_region[region]
            features[f'load_q_{region}_mvar'] = load_q_by_region[region]
            features[f'load_n_{region}'] = float(load_n_by_region[region])

        total_load_p = sum(load_p_by_region.values())
        total_load_q = sum(load_q_by_region.values())
        features['load_p_total_mw'] = float(total_load_p)
        features['load_q_total_mvar'] = float(total_load_q)

        all_p = [float(x) for x in res_load['p_mw'].values if pd.notna(x)]
        all_q = [float(x) for x in res_load['q_mvar'].values if pd.notna(x)]
        features['load_p_mean_mw'] = float(np.mean(all_p)) if all_p else 0.0
        features['load_q_mean_mvar'] = float(np.mean(all_q)) if all_q else 0.0
        features['load_p_std_mw'] = float(np.std(all_p)) if all_p else 0.0
    else:
        for region in ['r1', 'r2', 'r3']:
            features[f'load_p_{region}_mw'] = 0.0
            features[f'load_q_{region}_mvar'] = 0.0
            features[f'load_n_{region}'] = 0.0
        features['load_p_total_mw'] = 0.0
        features['load_q_total_mvar'] = 0.0
        features['load_p_mean_mw'] = 0.0
        features['load_q_mean_mvar'] = 0.0
        features['load_p_std_mw'] = 0.0

    # ---- Per-region generation aggregates by type ----
    # res_gen rows align 1:1 with gen_df rows
    if res_gen is not None and gen_df is not None:
        n_gens = min(len(res_gen), len(gen_df))

        # Pre-extract columns as numpy arrays for speed
        gen_bus_arr = np.array([gen_df['bus'].iat[i] for i in range(n_gens)], dtype=int)
        gen_name_arr = np.array([str(gen_df['name'].iat[i]) for i in range(n_gens)])
        gen_p_arr = np.array([float(res_gen['p_mw'].iat[i]) for i in range(n_gens)])
        gen_q_arr = np.array([float(res_gen['q_mvar'].iat[i]) for i in range(n_gens)])

        for region in ['r1', 'r2', 'r3']:
            r_buses = {b for b, m in bus_meta.items() if m['region'] == region}
            r_mask = np.array([b in r_buses for b in gen_bus_arr])

            for gtype in ['solar', 'wind', 'hydro', 'thermal']:
                gtype_set = _gen_types_for_category(gtype, gen_meta)
                gtype_mask = r_mask & np.array([gen_meta.get(gn, {}).get('type', '') in gtype_set for gn in gen_name_arr])
                gtype_p = float(np.sum(gen_p_arr[gtype_mask]))
                gtype_q = float(np.sum(gen_q_arr[gtype_mask]))
                gtype_n = int(np.sum(gtype_mask))
                features[f'gen_{gtype}_p_{region}_mw'] = gtype_p
                features[f'gen_{gtype}_q_{region}_mvar'] = gtype_q
                features[f'gen_{gtype}_n_{region}'] = float(gtype_n)

            total_p = float(np.sum(gen_p_arr[r_mask]))
            total_q = float(np.sum(gen_q_arr[r_mask]))
            total_n = int(np.sum(r_mask))
            features[f'gen_total_p_{region}_mw'] = total_p
            features[f'gen_total_q_{region}_mvar'] = total_q
            features[f'gen_total_n_{region}'] = float(total_n)

        all_p = gen_p_arr[~np.isnan(gen_p_arr)]
        all_q = gen_q_arr[~np.isnan(gen_q_arr)]
        features['gen_p_total_mw'] = float(np.sum(gen_p_arr))
        features['gen_q_total_mvar'] = float(np.sum(gen_q_arr))
        features['gen_p_mean_mw'] = float(np.mean(all_p)) if len(all_p) else 0.0
        features['gen_p_std_mw'] = float(np.std(all_p)) if len(all_p) else 0.0

        for gtype in ['solar', 'wind', 'hydro']:
            gtype_mask = np.array([gen_meta.get(gn, {}).get('type', '') == gtype for gn in gen_name_arr])
            features[f'gen_{gtype}_total_mw'] = float(np.sum(gen_p_arr[gtype_mask]))
    else:
        for region in ['r1', 'r2', 'r3']:
            for gtype in ['solar', 'wind', 'hydro', 'thermal']:
                features[f'gen_{gtype}_p_{region}_mw'] = 0.0
                features[f'gen_{gtype}_q_{region}_mvar'] = 0.0
                features[f'gen_{gtype}_n_{region}'] = 0.0
            features[f'gen_total_p_{region}_mw'] = 0.0
            features[f'gen_total_q_{region}_mvar'] = 0.0
            features[f'gen_total_n_{region}'] = 0.0
        features['gen_p_total_mw'] = 0.0
        features['gen_q_total_mvar'] = 0.0
        features['gen_p_mean_mw'] = 0.0
        features['gen_p_std_mw'] = 0.0
        for gtype in ['solar', 'wind', 'hydro']:
            features[f'gen_{gtype}_total_mw'] = 0.0

    features['net_interchange_mw'] = features['gen_p_total_mw'] - features['load_p_total_mw']

    # NOTE: Bus voltage statistics (system_vm_*, vm_mean_*, vm_std_*) are REMOVED
    # They were computed from the power flow SOLUTION (res_bus['vm_pu']), which is TARGET data.
    # Including them as features = DATA LEAKAGE.
    # The model would learn to predict voltages using the mean/std of voltages,
    # which is useless in real-world deployment where voltages are unknown before solving.

    # ---- Targets: vm_pu per bus (all 118 buses) ----
    if res_bus is not None and len(res_bus) == 118:
        for bus_idx in range(118):
            targets[f'vm_{bus_idx + 1}'] = float(res_bus.iloc[bus_idx]['vm_pu'])

    return features, targets


def parse_timestamp_from_filename(filename):
    """Extract datetime from sample filename like 2024_01_01_00_00_00.json."""
    ts_str = filename.replace('.json', '')
    return datetime.strptime(ts_str, '%Y_%m_%d_%H_%M_%S')


def run():
    parser = argparse.ArgumentParser(description='Prepare IEEE 118-Bus power flow data for LLMIP')
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES,
                        help='Number of hourly snapshots to sample (default: 200)')
    args = parser.parse_args()

    print("=" * 60)
    print("IEEE 118-Bus Power Flow Data Preparation")
    print("=" * 60)

    print("\n[1/5] Loading static bus and generator metadata...")
    bus_meta = extract_bus_metadata()
    gen_meta = extract_gen_metadata()
    load_meta = load_load_metadata()
    print(f"  Buses: {len(bus_meta)}, Gens: {len(gen_meta)}, Loads: {len(load_meta)}")

    print("\n[2/5] Finding all sample files...")
    all_files = sorted(SAMPLES_DIR.glob('*.json'))
    all_files = [f for f in all_files if 'Zone.Identifier' not in f.name]
    print(f"  Found {len(all_files)} sample files")

    if len(all_files) == 0:
        print("ERROR: No sample JSON files found in", SAMPLES_DIR)
        return

    all_timestamps = []
    for f in all_files:
        try:
            ts = parse_timestamp_from_filename(f.name)
            all_timestamps.append((ts, f))
        except ValueError:
            pass

    all_timestamps.sort()
    print(f"  Valid timestamps: {len(all_timestamps)}")
    print(f"  Range: {all_timestamps[0][0]} to {all_timestamps[-1][0]}")

    print(f"\n[3/5] Sampling {args.n_samples} snapshots (stratified by month)...")
    np.random.seed(RANDOM_STATE)
    n = args.n_samples

    by_month = {}
    for ts, fpath in all_timestamps:
        m = ts.month
        by_month.setdefault(m, []).append((ts, fpath))

    sampled = []
    per_month = max(1, n // 12)
    months = sorted(by_month.keys())
    for m in months:
        pool = by_month[m]
        k = min(per_month, len(pool))
        chosen = list(np.random.choice(len(pool), size=k, replace=False))
        sampled.extend([pool[i] for i in chosen])

    while len(sampled) < n:
        remaining = [x for x in all_timestamps if x not in sampled]
        if not remaining:
            break
        idx = np.random.randint(len(remaining))
        sampled.append(remaining[idx])

    np.random.shuffle(sampled)
    sampled = sampled[:n]
    print(f"  Sampled: {len(sampled)} snapshots")

    print("\n[4/5] Extracting features and targets from each snapshot...")
    all_features = []
    all_targets = []
    skipped = 0

    for i, (ts, fpath) in enumerate(sampled):
        if (i + 1) % 25 == 0:
            print(f"  Processing {i + 1}/{len(sampled)}...")

        feats, targs = extract_features_and_targets(fpath, bus_meta, gen_meta, load_meta, ts)

        if feats is None or targs is None or len(targs) != 118:
            skipped += 1
            continue

        feats['_timestamp'] = ts.isoformat()
        feats['_source_file'] = fpath.name
        all_features.append(feats)
        all_targets.append(targs)

    print(f"  Successfully extracted: {len(all_features)} snapshots")
    print(f"  Skipped (non-converged or missing): {skipped}")

    if len(all_features) < 20:
        print("ERROR: Too few valid snapshots. Check the data.")
        return

    feat_df = pd.DataFrame(all_features).set_index('_timestamp')
    targ_df = pd.DataFrame(all_targets)
    targ_df.index = feat_df.index

    feat_df = feat_df.drop(columns=['_source_file'], errors='ignore')

    train_size = int(len(feat_df) * 0.8)
    train_feats = feat_df.iloc[:train_size]
    test_feats = feat_df.iloc[train_size:]
    train_targ = targ_df.iloc[:train_size]
    test_targ = targ_df.iloc[train_size:]

    print(f"\n  Train: {len(train_feats)} samples, Test: {len(test_feats)} samples")

    train_feats.to_csv(PREPARED_DIR / 'llmip_train_features.csv')
    test_feats.to_csv(PREPARED_DIR / 'llmip_test_features.csv')
    train_targ.to_csv(PREPARED_DIR / 'llmip_train_targets.csv')
    test_targ.to_csv(PREPARED_DIR / 'llmip_test_targets.csv')

    print(f"\n[5/5] Saving summary statistics...")
    vm_cols = [c for c in targ_df.columns if c.startswith('vm_')]
    stats = {
        'dataset': 'IEEE 118-Bus Power Flow (evgenytsydenov/ieee118_power_flow_data)',
        'source': 'https://github.com/evgenytsydenov/ieee118_power_flow_data',
        'n_train': int(len(train_feats)),
        'n_test': int(len(test_feats)),
        'n_features': int(len(feat_df.columns)),
        'n_buses': len(vm_cols),
        'feature_names': list(feat_df.columns),
        'target_names': list(targ_df.columns),
        'generator_buses': [int(k) + 1 for k, v in bus_meta.items()],
        'slack_bus': int([k for k, v in bus_meta.items() if v.get('is_slack', False)][0]) + 1,
        'regions': {
            'r1': sum(1 for v in bus_meta.values() if v['region'] == 'r1'),
            'r2': sum(1 for v in bus_meta.values() if v['region'] == 'r2'),
            'r3': sum(1 for v in bus_meta.values() if v['region'] == 'r3'),
        },
        'vm_pu_stats': {
            'overall_mean': float(targ_df[vm_cols].values.mean()),
            'overall_std': float(targ_df[vm_cols].values.std()),
            'overall_min': float(targ_df[vm_cols].values.min()),
            'overall_max': float(targ_df[vm_cols].values.max()),
        },
        'timestamp_range': {
            'start': str(feat_df.index[0]),
            'end': str(feat_df.index[-1]),
        }
    }

    with open(PREPARED_DIR / 'llmip_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Train features: {PREPARED_DIR / 'llmip_train_features.csv'}")
    print(f"  Train targets:  {PREPARED_DIR / 'llmip_train_targets.csv'}")
    print(f"  Test features:  {PREPARED_DIR / 'llmip_test_features.csv'}")
    print(f"  Test targets:   {PREPARED_DIR / 'llmip_test_targets.csv'}")
    print(f"  Stats:          {PREPARED_DIR / 'llmip_stats.json'}")
    print(f"\n  Features: {stats['n_features']}")
    print(f"  Targets:  {stats['n_buses']} bus voltages (vm_pu)")
    print(f"  Train:    {stats['n_train']} samples")
    print(f"  Test:     {stats['n_test']} samples")
    print(f"\n  vm_pu range: {stats['vm_pu_stats']['overall_min']:.4f} - "
          f"{stats['vm_pu_stats']['overall_max']:.4f} pu")


if __name__ == '__main__':
    run()
