import pandas as pd
import numpy as np
from collections import defaultdict

INPUT_PATH = 'fixeddata.csv'
ISSUES_OUT = 'data_issues.csv'
TEMPLATE_OUT = 'clean_template.csv'

VALID_RANGES = {
    'height_inches': (60, 90),
    'weight_lbs': (140, 310),
    'ppg': (0, 35),
    'apg': (0, 12),
    'rpg': (0, 17),
    'spg': (0, 5),
    'bpg': (0, 6),
    'mpg': (0, 40),
    'topg': (0, 8),
    'fpg': (0, 5),
    'fg_pct': (0.2, 0.85),
    'three_pt_pct': (0.0, 1.0),
    'ft_pct': (0.0, 1.0),
    'efg_pct': (0.2, 0.85),
    'two_pt_pct': (0.2, 0.85),
    'ws': (-5, 35),
    'ws_40': (-0.25, 0.5),
    'ows': (-5, 25),
    'dws': (-2, 20),
    'ts_pct': (0.3, 0.85),
    'tov_pct': (0, 40),
    'ftr': (0, 1.2),
    'three_pt_attempt_rate': (0, 1.0),
    'college_seasons': (1, 5),
    'recruit_rank': (1, 101),
}

IMPUTED_SIGNATURES = {
    'ppg': [13.713541994668068, 10.673181506500002],
    'apg': [3.1496565537602943, 0.9557534450805085],
    'rpg': [4.163283735327731, 6.7030888885],
    'spg': [1.2694005429157564, 0.6243440892033898],
    'bpg': [0.4009285890569748, 1.8517245503508475],
}

Z_SCORE_THRESHOLD = 4.0
IDENTITY_COLS = ['player_name', 'draft_year', 'pick_number', 'college', 'position']

def audit() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {INPUT_PATH}\n")

    issues = []

    print("=" * 70)
    print("1. MISSING VALUES")
    print("=" * 70)
    for col in df.columns:
        missing_mask = df[col].isna()
        n_missing = missing_mask.sum()
        if n_missing == 0:
            continue
        print(f"  {col}: {n_missing} missing")
        for idx in df.index[missing_mask]:
            issues.append({
                'row_idx': idx,
                'player_name': df.at[idx, 'player_name'],
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': col,
                'value': '',
                'issue': 'missing',
            })

    print("\n" + "=" * 70)
    print("2. VALUES OUTSIDE VALID RANGES")
    print("=" * 70)
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        out_mask = numeric.notna() & ((numeric < lo) | (numeric > hi))
        n_out = out_mask.sum()
        if n_out == 0:
            continue
        print(f"\n  {col} (valid: {lo}-{hi}) — {n_out} violations:")
        for idx in df.index[out_mask]:
            val = df.at[idx, col]
            print(f"    row {idx:4d}  {df.at[idx, 'player_name']:30s} ({int(df.at[idx, 'draft_year'])}, {df.at[idx, 'college']}):  {val}")
            issues.append({
                'row_idx': idx,
                'player_name': df.at[idx, 'player_name'],
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': col,
                'value': val,
                'issue': f'out_of_range [{lo},{hi}]',
            })

    print("\n" + "=" * 70)
    print(f"3. EXTREME OUTLIERS (|z| > {Z_SCORE_THRESHOLD})")
    print("=" * 70)
    for col in VALID_RANGES:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        lo, hi = VALID_RANGES[col]

        in_range = numeric[(numeric >= lo) & (numeric <= hi)]
        if len(in_range) < 30:
            continue
        mu, sd = in_range.mean(), in_range.std()
        if sd == 0:
            continue
        z = (numeric - mu) / sd
        mask = numeric.notna() & (z.abs() > Z_SCORE_THRESHOLD) & (numeric >= lo) & (numeric <= hi)
        if mask.sum() == 0:
            continue
        print(f"\n  {col} (mean={mu:.3f}, std={sd:.3f}) — {mask.sum()} extreme outliers:")
        for idx in df.index[mask]:
            val = df.at[idx, col]
            print(f"    row {idx:4d}  {df.at[idx, 'player_name']:30s}  value={val:.3f}  z={z[idx]:+.2f}")
            issues.append({
                'row_idx': idx,
                'player_name': df.at[idx, 'player_name'],
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': col,
                'value': val,
                'issue': f'extreme_outlier (z={z[idx]:+.2f})',
            })

    print("\n" + "=" * 70)
    print("4. SUSPECTED MEAN-IMPUTED ROWS")
    print("=" * 70)
    imputed_rows = set()
    for col, signatures in IMPUTED_SIGNATURES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        for sig in signatures:
            mask = np.isclose(numeric, sig, atol=1e-9)
            if mask.sum() == 0:
                continue
            for idx in df.index[mask]:
                imputed_rows.add(idx)
                issues.append({
                    'row_idx': idx,
                    'player_name': df.at[idx, 'player_name'],
                    'draft_year': df.at[idx, 'draft_year'],
                    'college': df.at[idx, 'college'],
                    'column': col,
                    'value': sig,
                    'issue': f'mean_imputed_placeholder (matches global mean exactly)',
                })
    print(f"\n  {len(imputed_rows)} rows contain placeholder values:")
    for idx in sorted(imputed_rows):
        print(f"    row {idx:4d}  {df.at[idx, 'player_name']:30s}  ({int(df.at[idx, 'draft_year'])}, {df.at[idx, 'college']})")

    print("\n" + "=" * 70)
    print("5. SUSPECTED COLUMN MISALIGNMENT (impossible combinations)")
    print("=" * 70)

    for idx in df.index:
        mpg = pd.to_numeric(df.at[idx, 'mpg'], errors='coerce')
        fgp = pd.to_numeric(df.at[idx, 'fg_pct'], errors='coerce')
        if pd.notna(mpg) and pd.notna(fgp) and mpg == fgp and mpg > 10:
            print(f"  row {idx:4d}  {df.at[idx, 'player_name']:30s}  fg_pct={fgp} == mpg={mpg}")
            issues.append({
                'row_idx': idx,
                'player_name': df.at[idx, 'player_name'],
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': 'fg_pct',
                'value': fgp,
                'issue': 'misaligned: fg_pct == mpg',
            })

    for idx in df.index:
        spg = pd.to_numeric(df.at[idx, 'spg'], errors='coerce')
        bpg = pd.to_numeric(df.at[idx, 'bpg'], errors='coerce')
        if pd.notna(spg) and spg > 5:
            print(f"  row {idx:4d}  {df.at[idx, 'player_name']:30s}  spg={spg} (impossibly high)")
            issues.append({
                'row_idx': idx,
                'player_name': df.at[idx, 'player_name'],
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': 'spg',
                'value': spg,
                'issue': 'impossibly_high_steals',
            })

    print("\n" + "=" * 70)
    print("6. CORRUPTED PLAYER NAMES")
    print("=" * 70)
    for idx in df.index:
        name = str(df.at[idx, 'player_name'])
        if 'http' in name.lower() or len(name) > 50 or '@' in name:
            print(f"  row {idx:4d}  {name[:80]}")
            issues.append({
                'row_idx': idx,
                'player_name': name,
                'draft_year': df.at[idx, 'draft_year'],
                'college': df.at[idx, 'college'],
                'column': 'player_name',
                'value': name,
                'issue': 'corrupted_name',
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    issues_df = pd.DataFrame(issues)
    if len(issues_df):
        print(f"  Total flagged cells: {len(issues_df)}")
        print(f"  Unique rows affected: {issues_df['row_idx'].nunique()}")
        print(f"\n  Issue breakdown:")
        for issue_type, count in issues_df['issue'].str.split(' ').str[0].value_counts().items():
            print(f"    {issue_type}: {count}")

        issues_df.sort_values(['row_idx', 'column']).to_csv(ISSUES_OUT, index=False)
        print(f"\n  Full issue list written to: {ISSUES_OUT}")
    else:
        print("  No issues detected.")

    template = df.copy()
    for _, row in issues_df.iterrows():
        col = row['column']
        idx = row['row_idx']
        if col in template.columns and col not in IDENTITY_COLS:
            template.at[idx, col] = ''
    template.to_csv(TEMPLATE_OUT, index=False)
    print(f"  Clean-fill template written to: {TEMPLATE_OUT}")
    print(f"    (flagged cells are emptied — fill them in manually, then save)\n")

if __name__ == '__main__':
    audit()
