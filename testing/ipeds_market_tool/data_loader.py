"""
Data loading and processing module for IPEDS Market Targeting Tool.
Handles CSV ingestion, merging, derived metrics, and data quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from typing import Dict, Optional, Tuple, List
from config import (
    SECTOR_LABELS, CONTROL_LABELS, ICLEVEL_LABELS, LOCALE_LABELS,
    BEA_REGION_LABELS, STATE_TO_REGION, DE_LEVEL_CODES
)


@st.cache_data(ttl=3600)
def load_csv_cached(filepath: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load CSV with caching for performance."""
    try:
        # Handle potential Windows line endings
        df = pd.read_csv(filepath, usecols=usecols, low_memory=False, encoding='utf-8')
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return pd.DataFrame()


def load_directory_file(filepath: str) -> Tuple[pd.DataFrame, bool]:
    """
    Load the HD (directory) file with geographic information.
    Returns (dataframe, success_flag).
    """
    required_cols = ['UNITID', 'INSTNM', 'CITY', 'STABBR']
    optional_cols = ['ZIP', 'LATITUDE', 'LONGITUD', 'SECTOR', 'CONTROL', 
                     'ICLEVEL', 'LOCALE', 'HLOFFER', 'UGOFFER', 'GROFFER',
                     'HBCU', 'INSTSIZE', 'OBEREG', 'LANDGRNT']
    
    try:
        df = load_csv_cached(filepath)
        
        # Check for required columns
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            st.warning(f"Directory file missing required columns: {missing_required}")
            return pd.DataFrame(), False
        
        # Select available columns
        available_cols = [c for c in required_cols + optional_cols if c in df.columns]
        df = df[available_cols].copy()
        
        # Add region based on state
        if 'STABBR' in df.columns:
            df['REGION'] = df['STABBR'].map(STATE_TO_REGION).fillna('Unknown')
        
        # Add label columns for display
        if 'SECTOR' in df.columns:
            df['SECTOR_LABEL'] = df['SECTOR'].map(SECTOR_LABELS).fillna('Unknown')
        if 'CONTROL' in df.columns:
            df['CONTROL_LABEL'] = df['CONTROL'].map(CONTROL_LABELS).fillna('Unknown')
        if 'ICLEVEL' in df.columns:
            df['ICLEVEL_LABEL'] = df['ICLEVEL'].map(ICLEVEL_LABELS).fillna('Unknown')
        if 'LOCALE' in df.columns:
            df['LOCALE_LABEL'] = df['LOCALE'].map(LOCALE_LABELS).fillna('Unknown')
        if 'OBEREG' in df.columns:
            df['OBEREG_LABEL'] = df['OBEREG'].map(BEA_REGION_LABELS).fillna('Unknown')
        
        return df, True
        
    except Exception as e:
        st.error(f"Error loading directory file: {str(e)}")
        return pd.DataFrame(), False


@st.cache_data(ttl=3600)
def load_drvc(filepath: str) -> pd.DataFrame:
    """Load degree/certificate offerings data."""
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Create summary flags
    df['HAS_ASSOCIATES'] = (df.get('ASCDEG', 0) > 0).astype(int)
    df['HAS_BACHELORS'] = (df.get('BASDEG', 0) > 0).astype(int)
    df['HAS_MASTERS'] = (df.get('MASDEG', 0) > 0).astype(int)
    df['HAS_DOCTORAL'] = ((df.get('DOCDEGRS', 0) > 0) | 
                          (df.get('DOCDEGPP', 0) > 0) | 
                          (df.get('DOCDEGOT', 0) > 0)).astype(int)
    df['HAS_CERTIFICATES'] = ((df.get('CERT1', 0) > 0) | 
                              (df.get('CERT1A', 0) > 0) |
                              (df.get('CERT1B', 0) > 0) |
                              (df.get('CERT2', 0) > 0) |
                              (df.get('CERT4', 0) > 0)).astype(int)
    df['HAS_POSTBACC_CERT'] = (df.get('PBACERT', 0) > 0).astype(int)
    df['HAS_POSTMASTERS_CERT'] = (df.get('PMACERT', 0) > 0).astype(int)
    
    # Total graduate awards
    df['TOTAL_GRAD_AWARDS'] = (df.get('MASDEG', 0).fillna(0) + 
                               df.get('DOCDEGRS', 0).fillna(0) + 
                               df.get('DOCDEGPP', 0).fillna(0) +
                               df.get('DOCDEGOT', 0).fillna(0) +
                               df.get('PBACERT', 0).fillna(0) +
                               df.get('PMACERT', 0).fillna(0))
    
    # Total undergrad awards
    df['TOTAL_UG_AWARDS'] = (df.get('ASCDEG', 0).fillna(0) + 
                             df.get('BASDEG', 0).fillna(0) +
                             df.get('CERT1', 0).fillna(0) +
                             df.get('CERT1A', 0).fillna(0) +
                             df.get('CERT1B', 0).fillna(0) +
                             df.get('CERT2', 0).fillna(0) +
                             df.get('CERT4', 0).fillna(0))
    
    return df


@st.cache_data(ttl=3600)
def load_drvef(filepath: str) -> pd.DataFrame:
    """Load enrollment data with derived metrics."""
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Calculate graduate percentage
    df['GRAD_PCT'] = np.where(
        df['ENRTOT'] > 0,
        (df.get('EFGRAD', 0) / df['ENRTOT'] * 100).round(1),
        0
    )
    
    # Part-time percentage
    df['PT_PCT'] = np.where(
        df['ENRTOT'] > 0,
        (df.get('ENRPT', 0) / df['ENRTOT'] * 100).round(1),
        0
    )
    
    # Mobility index (out-of-state + foreign)
    df['MOBILITY_INDEX'] = (df.get('RMOUSTTP', 0).fillna(0) + 
                            df.get('RMFRGNCP', 0).fillna(0))
    
    return df


@st.cache_data(ttl=3600)
def load_distance_ed(filepath: str) -> pd.DataFrame:
    """
    Load and aggregate distance education data.
    Returns one row per UNITID with totals and breakdowns.
    """
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Filter to total enrollment level (EFDELEV=1) for institution-level stats
    df_total = df[df['EFDELEV'] == 1].copy()
    
    # Also get undergraduate (EFDELEV=2) and graduate (EFDELEV=12) breakdowns
    df_ug = df[df['EFDELEV'] == 2][['UNITID', 'EFDETOT', 'EFDEEXC', 'EFDESOM', 'EFDENON']].copy()
    df_ug.columns = ['UNITID', 'DE_UG_TOTAL', 'DE_UG_EXCLUSIVE', 'DE_UG_SOME', 'DE_UG_NONE']
    
    df_grad = df[df['EFDELEV'] == 12][['UNITID', 'EFDETOT', 'EFDEEXC', 'EFDESOM', 'EFDENON']].copy()
    df_grad.columns = ['UNITID', 'DE_GRAD_TOTAL', 'DE_GRAD_EXCLUSIVE', 'DE_GRAD_SOME', 'DE_GRAD_NONE']
    
    # Merge
    result = df_total[['UNITID', 'EFDETOT', 'EFDEEXC', 'EFDESOM', 'EFDENON']].copy()
    result.columns = ['UNITID', 'DE_TOTAL', 'DE_EXCLUSIVE', 'DE_SOME', 'DE_NONE']
    
    result = result.merge(df_ug, on='UNITID', how='left')
    result = result.merge(df_grad, on='UNITID', how='left')
    
    # Calculate percentages
    for prefix in ['', '_UG', '_GRAD']:
        total_col = f'DE{prefix}_TOTAL'
        if total_col in result.columns:
            result[f'DE{prefix}_EXCLUSIVE_PCT'] = np.where(
                result[total_col] > 0,
                (result.get(f'DE{prefix}_EXCLUSIVE', 0) / result[total_col] * 100).round(1),
                0
            )
            result[f'DE{prefix}_SOME_PCT'] = np.where(
                result[total_col] > 0,
                (result.get(f'DE{prefix}_SOME', 0) / result[total_col] * 100).round(1),
                0
            )
    
    # Online intensity score (0-100)
    result['ONLINE_INTENSITY'] = (
        result.get('DE_EXCLUSIVE_PCT', 0).fillna(0) * 1.0 +
        result.get('DE_SOME_PCT', 0).fillna(0) * 0.5
    ).clip(0, 100)
    
    return result


@st.cache_data(ttl=3600)
def load_cost(filepath: str) -> pd.DataFrame:
    """Load cost data with key tuition/fee columns."""
    # Only load columns we need
    cols_to_load = ['UNITID', 'TUITION1', 'TUITION2', 'TUITION3', 
                    'FEE1', 'FEE2', 'FEE3', 'ROOMAMT', 'BOARDAMT', 
                    'RMBRDAMT', 'TUITVARY', 'TUITION5', 'TUITION6', 'TUITION7']
    
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Select only available columns
    available = [c for c in cols_to_load if c in df.columns]
    df = df[available].copy()
    
    # Create combined tuition + fees columns
    df['TOTAL_COST_IN_STATE'] = df.get('TUITION2', 0).fillna(0) + df.get('FEE2', 0).fillna(0)
    df['TOTAL_COST_OUT_STATE'] = df.get('TUITION3', 0).fillna(0) + df.get('FEE3', 0).fillna(0)
    
    # Identify graduate tuition if available (TUITION5 is often grad in-state)
    if 'TUITION5' in df.columns:
        df['GRAD_TUITION_IN_STATE'] = df['TUITION5'].fillna(0)
    if 'TUITION6' in df.columns:
        df['GRAD_TUITION_OUT_STATE'] = df['TUITION6'].fillna(0)
    
    return df


@st.cache_data(ttl=3600)
def load_completions(filepath: str) -> pd.DataFrame:
    """Load completions data (c2024_b)."""
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Key columns
    cols = ['UNITID', 'CSTOTLT', 'CSTOTLM', 'CSTOTLW']
    available = [c for c in cols if c in df.columns]
    
    return df[available].copy() if available else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_flags(filepath: str) -> pd.DataFrame:
    """Load data quality flags."""
    df = load_csv_cached(filepath)
    if df.empty:
        return df
    
    # Create a simplified data quality indicator
    # IMP columns indicate imputation; higher values = more imputation
    imp_cols = [c for c in df.columns if c.startswith('IMP_')]
    
    if imp_cols:
        # Count how many surveys have imputed values
        df['IMPUTATION_COUNT'] = df[imp_cols].apply(
            lambda row: sum(1 for v in row if v not in [-2, -1, 0]), axis=1
        )
        df['DATA_QUALITY'] = np.where(
            df['IMPUTATION_COUNT'] == 0, 'High',
            np.where(df['IMPUTATION_COUNT'] <= 3, 'Medium', 'Low')
        )
    else:
        df['IMPUTATION_COUNT'] = 0
        df['DATA_QUALITY'] = 'Unknown'
    
    return df[['UNITID', 'IMPUTATION_COUNT', 'DATA_QUALITY']]


@st.cache_data(ttl=3600)
def load_effy_summary(filepath: str) -> pd.DataFrame:
    """
    Load effy file and create summary statistics.
    This file is large so we aggregate to institution level.
    """
    # Read in chunks for large file
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=50000, low_memory=False):
        # Filter to total level (EFFYLEV=1 or EFFYALEV=1)
        filtered = chunk[(chunk['EFFYLEV'] == 1) | (chunk['EFFYALEV'] == 1)]
        if not filtered.empty:
            chunks.append(filtered)
    
    if not chunks:
        return pd.DataFrame()
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Aggregate by UNITID - take first row per institution at total level
    df = df.groupby('UNITID').first().reset_index()
    
    # Select key demographic columns
    key_cols = ['UNITID', 'EFYTOTLT', 'EFYTOTLM', 'EFYTOTLW',
                'EFYWHITT', 'EFYBKAAT', 'EFYHISPT', 'EFYASIAT',
                'EFYNRALT']  # Nonresident alien
    
    available = [c for c in key_cols if c in df.columns]
    return df[available].copy() if available else pd.DataFrame()


def merge_all_data(data_dir: str, hd_filepath: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and merge all data files into a single analysis dataset.
    Returns (merged_df, merge_stats).
    """
    merge_stats = {
        'files_loaded': [],
        'unmatched_ids': {},
        'total_institutions': 0,
        'has_geography': False
    }
    
    # Start with enrollment as base (most institutions have this)
    drvef = load_drvef(f"{data_dir}/drvef2024.csv")
    if drvef.empty:
        st.error("Could not load enrollment data (drvef2024.csv)")
        return pd.DataFrame(), merge_stats
    
    base_df = drvef.copy()
    merge_stats['files_loaded'].append('drvef2024')
    merge_stats['total_institutions'] = len(base_df)
    
    # Load directory if available
    if hd_filepath:
        hd_df, success = load_directory_file(hd_filepath)
        if success and not hd_df.empty:
            pre_merge = len(base_df)
            base_df = base_df.merge(hd_df, on='UNITID', how='left')
            merge_stats['files_loaded'].append('hd2024 (directory)')
            merge_stats['has_geography'] = True
            merge_stats['unmatched_ids']['hd2024'] = pre_merge - base_df['INSTNM'].notna().sum()
    
    # Merge degree/certificate data
    drvc = load_drvc(f"{data_dir}/drvc2024.csv")
    if not drvc.empty:
        pre_merge = len(base_df)
        base_df = base_df.merge(drvc, on='UNITID', how='left')
        merge_stats['files_loaded'].append('drvc2024')
        merge_stats['unmatched_ids']['drvc2024'] = pre_merge - base_df['HAS_MASTERS'].notna().sum()
    
    # Merge distance education
    de_df = load_distance_ed(f"{data_dir}/ef2024a_dist.csv")
    if not de_df.empty:
        pre_merge = len(base_df)
        base_df = base_df.merge(de_df, on='UNITID', how='left')
        merge_stats['files_loaded'].append('ef2024a_dist')
        merge_stats['unmatched_ids']['ef2024a_dist'] = pre_merge - base_df['DE_TOTAL'].notna().sum()
    
    # Merge cost data
    cost = load_cost(f"{data_dir}/cost1_2024.csv")
    if not cost.empty:
        pre_merge = len(base_df)
        base_df = base_df.merge(cost, on='UNITID', how='left')
        merge_stats['files_loaded'].append('cost1_2024')
        merge_stats['unmatched_ids']['cost1_2024'] = pre_merge - base_df['TUITION2'].notna().sum()
    
    # Merge flags
    flags = load_flags(f"{data_dir}/flags2024.csv")
    if not flags.empty:
        base_df = base_df.merge(flags, on='UNITID', how='left')
        merge_stats['files_loaded'].append('flags2024')
    
    # Create derived metrics
    base_df = create_derived_metrics(base_df)
    
    return base_df, merge_stats


def create_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite scores and derived metrics."""
    
    # Grad Focus Index (0-100)
    # Based on: has masters/doctoral, grad enrollment share, grad awards
    grad_factors = []
    
    if 'HAS_MASTERS' in df.columns:
        grad_factors.append(df['HAS_MASTERS'] * 25)
    if 'HAS_DOCTORAL' in df.columns:
        grad_factors.append(df['HAS_DOCTORAL'] * 25)
    if 'GRAD_PCT' in df.columns:
        grad_factors.append(df['GRAD_PCT'].clip(0, 50))  # Cap at 50 points
    
    if grad_factors:
        df['GRAD_FOCUS_INDEX'] = sum(grad_factors).clip(0, 100)
    else:
        df['GRAD_FOCUS_INDEX'] = 0
    
    # Online Intensity already calculated in load_distance_ed
    if 'ONLINE_INTENSITY' not in df.columns:
        df['ONLINE_INTENSITY'] = 0
    
    # Mobility Index already in drvef
    if 'MOBILITY_INDEX' not in df.columns:
        df['MOBILITY_INDEX'] = 0
    
    # Enrollment Scale (normalized 0-100)
    if 'FTE' in df.columns:
        df['ENROLLMENT_SCALE'] = pd.qcut(
            df['FTE'].fillna(0).clip(lower=1), 
            q=10, 
            labels=range(10, 101, 10),
            duplicates='drop'
        ).astype(float).fillna(10)
    else:
        df['ENROLLMENT_SCALE'] = 50
    
    # Cost Accessibility (inverse - lower cost = higher score)
    if 'TOTAL_COST_IN_STATE' in df.columns:
        cost_col = df['TOTAL_COST_IN_STATE'].fillna(df['TOTAL_COST_IN_STATE'].median())
        cost_percentile = cost_col.rank(pct=True)
        df['COST_ACCESSIBILITY'] = ((1 - cost_percentile) * 100).round(1)
    else:
        df['COST_ACCESSIBILITY'] = 50
    
    return df


def calculate_market_fit_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Calculate weighted Market Fit Score.
    Weights should sum to 1.0.
    """
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    norm_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Map weight keys to columns
    weight_to_col = {
        'grad_focus': 'GRAD_FOCUS_INDEX',
        'online_intensity': 'ONLINE_INTENSITY',
        'mobility_index': 'MOBILITY_INDEX',
        'enrollment_scale': 'ENROLLMENT_SCALE',
        'cost_accessibility': 'COST_ACCESSIBILITY'
    }
    
    score = pd.Series(0.0, index=df.index)
    
    for weight_key, col_name in weight_to_col.items():
        if col_name in df.columns and weight_key in norm_weights:
            # Normalize column to 0-100 if not already
            col_vals = df[col_name].fillna(0).clip(0, 100)
            score += col_vals * norm_weights[weight_key]
    
    return score.round(1)


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply user-selected filters to the dataframe."""
    filtered = df.copy()
    
    # Geography filters (if available)
    if 'states' in filters and filters['states'] and 'STABBR' in filtered.columns:
        filtered = filtered[filtered['STABBR'].isin(filters['states'])]
    
    if 'regions' in filters and filters['regions'] and 'REGION' in filtered.columns:
        filtered = filtered[filtered['REGION'].isin(filters['regions'])]
    
    # Institution type filters
    if 'control' in filters and filters['control'] and 'CONTROL' in filtered.columns:
        filtered = filtered[filtered['CONTROL'].isin(filters['control'])]
    
    if 'iclevel' in filters and filters['iclevel'] and 'ICLEVEL' in filtered.columns:
        filtered = filtered[filtered['ICLEVEL'].isin(filters['iclevel'])]
    
    if 'sector' in filters and filters['sector'] and 'SECTOR' in filtered.columns:
        filtered = filtered[filtered['SECTOR'].isin(filters['sector'])]
    
    if 'locale' in filters and filters['locale'] and 'LOCALE' in filtered.columns:
        filtered = filtered[filtered['LOCALE'].isin(filters['locale'])]
    
    # Degree offerings filters
    if 'award_levels' in filters and filters['award_levels']:
        award_mask = pd.Series(False, index=filtered.index)
        if 'Associates' in filters['award_levels'] and 'HAS_ASSOCIATES' in filtered.columns:
            award_mask |= filtered['HAS_ASSOCIATES'] == 1
        if 'Bachelors' in filters['award_levels'] and 'HAS_BACHELORS' in filtered.columns:
            award_mask |= filtered['HAS_BACHELORS'] == 1
        if 'Masters' in filters['award_levels'] and 'HAS_MASTERS' in filtered.columns:
            award_mask |= filtered['HAS_MASTERS'] == 1
        if 'Doctoral' in filters['award_levels'] and 'HAS_DOCTORAL' in filtered.columns:
            award_mask |= filtered['HAS_DOCTORAL'] == 1
        if 'Certificates' in filters['award_levels'] and 'HAS_CERTIFICATES' in filtered.columns:
            award_mask |= filtered['HAS_CERTIFICATES'] == 1
        filtered = filtered[award_mask]
    
    # Distance education filter
    if 'distance_ed' in filters and filters['distance_ed']:
        de_mask = pd.Series(False, index=filtered.index)
        if 'Exclusive DE' in filters['distance_ed'] and 'PCTDEEXC' in filtered.columns:
            de_mask |= filtered['PCTDEEXC'] > 50
        if 'Some DE' in filters['distance_ed'] and 'PCTDESOM' in filtered.columns:
            de_mask |= (filtered['PCTDESOM'] > 20) & (filtered.get('PCTDEEXC', 0) <= 50)
        if 'No DE' in filters['distance_ed'] and 'PCTDENON' in filtered.columns:
            de_mask |= filtered['PCTDENON'] > 80
        filtered = filtered[de_mask]
    
    # Enrollment size filters
    if 'min_enrollment' in filters and filters['min_enrollment'] and 'ENRTOT' in filtered.columns:
        filtered = filtered[filtered['ENRTOT'] >= filters['min_enrollment']]
    
    if 'max_enrollment' in filters and filters['max_enrollment'] and 'ENRTOT' in filtered.columns:
        filtered = filtered[filtered['ENRTOT'] <= filters['max_enrollment']]
    
    # Data quality filter
    if 'hide_low_quality' in filters and filters['hide_low_quality'] and 'DATA_QUALITY' in filtered.columns:
        filtered = filtered[filtered['DATA_QUALITY'] != 'Low']
    
    return filtered


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by state."""
    if 'STABBR' not in df.columns:
        return pd.DataFrame()
    
    agg_dict = {
        'UNITID': 'count',
        'ENRTOT': 'sum',
        'FTE': 'sum',
        'EFGRAD': 'sum',
    }
    
    # Add optional columns
    if 'DE_EXCLUSIVE' in df.columns:
        agg_dict['DE_EXCLUSIVE'] = 'sum'
    if 'GRAD_FOCUS_INDEX' in df.columns:
        agg_dict['GRAD_FOCUS_INDEX'] = 'mean'
    if 'ONLINE_INTENSITY' in df.columns:
        agg_dict['ONLINE_INTENSITY'] = 'mean'
    if 'MOBILITY_INDEX' in df.columns:
        agg_dict['MOBILITY_INDEX'] = 'mean'
    if 'MARKET_FIT_SCORE' in df.columns:
        agg_dict['MARKET_FIT_SCORE'] = 'mean'
    
    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns or k == 'UNITID'}
    
    summary = df.groupby('STABBR').agg(agg_dict).reset_index()
    summary.columns = ['State', 'Institution_Count', 'Total_Enrollment', 'Total_FTE', 
                       'Total_Grad_Enrollment'] + [c for c in summary.columns[5:]]
    
    return summary.round(1)
