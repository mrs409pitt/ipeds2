"""
IPEDS Market Targeting Tool
A decision-support application for higher education paid media targeting.

Main Streamlit application file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Local imports
from data_loader import (
    merge_all_data, apply_filters, calculate_market_fit_score,
    get_state_summary, load_directory_file
)
from visualizations import (
    create_kpi_card, create_enrollment_histogram, create_sector_breakdown,
    create_de_breakdown, create_grad_ug_comparison, create_mobility_scatter,
    create_state_choropleth, create_institution_map, create_market_fit_components,
    create_state_ranking_bar, create_tuition_boxplot, create_enrollment_by_control,
    create_score_distribution, style_dataframe, generate_key_takeaways,
    format_strategy_summary, COLORS
)
from config import (
    COLUMN_LABELS, COLUMN_TOOLTIPS, SECTOR_LABELS, CONTROL_LABELS,
    ICLEVEL_LABELS, LOCALE_LABELS, REGION_LIST, DEFAULT_SCORE_WEIGHTS,
    WEIGHT_PRESETS, FILTER_PRESETS, MARKET_TAGS
)

# Page config
st.set_page_config(
    page_title="IPEDS Market Targeting Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E4057;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .callout-info {
        background-color: #E7F5FF;
        border-left: 4px solid #54C6EB;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .callout-warning {
        background-color: #FFF9E6;
        border-left: 4px solid #F4A024;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'merge_stats' not in st.session_state:
    st.session_state.merge_stats = {}
if 'weights' not in st.session_state:
    st.session_state.weights = DEFAULT_SCORE_WEIGHTS.copy()
if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'has_geography' not in st.session_state:
    st.session_state.has_geography = False


@st.cache_data(ttl=3600)
def load_data_cached(data_dir: str, hd_filepath: str = None):
    """Load and cache all data."""
    return merge_all_data(data_dir, hd_filepath)


def render_data_loader():
    """Render data loading interface."""
    st.markdown('<p class="main-header">🎯 IPEDS Market Targeting Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Geographic targeting intelligence for higher education paid media</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Data Configuration")
        data_dir = st.text_input("Data Directory Path", value="data",
                                 help="Path to folder containing IPEDS CSV files")
        
        st.markdown("#### Geographic Data (Optional)")
        st.markdown("""
        <div class="callout-info">
        <strong>📍 Enable Geographic Features</strong><br>
        The HD (directory) file enables map visualizations and state/city filters.
        Required columns: UNITID, INSTNM, CITY, STABBR, LATITUDE, LONGITUDE.
        </div>
        """, unsafe_allow_html=True)
        
        hd_option = st.radio("HD File Source",
                             ["Use file from data directory", "Upload HD file", "Skip (demo mode)"],
                             horizontal=True)
        
        hd_filepath = None
        if hd_option == "Use file from data directory":
            hd_filepath = f"{data_dir}/hd2024.csv"
            if not Path(hd_filepath).exists():
                st.warning(f"HD file not found at {hd_filepath}")
                hd_filepath = None
        elif hd_option == "Upload HD file":
            uploaded = st.file_uploader("Upload HD2024.csv", type=['csv'])
            if uploaded:
                temp_path = "/tmp/hd2024_uploaded.csv"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded.getvalue())
                hd_filepath = temp_path
    
    with col2:
        st.markdown("### Required Files")
        st.markdown("""
        - drvef2024.csv (enrollment)
        - drvc2024.csv (degrees)
        - ef2024a_dist.csv (distance ed)
        - cost1_2024.csv (tuition)
        - flags2024.csv (data quality)
        
        **Optional:** hd2024.csv (geography)
        """)
    
    if st.button("🚀 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading and processing data..."):
            df, stats = load_data_cached(data_dir, hd_filepath)
            
            if not df.empty:
                st.session_state.df = df
                st.session_state.merge_stats = stats
                st.session_state.data_loaded = True
                st.session_state.has_geography = stats.get('has_geography', False)
                
                st.session_state.df['MARKET_FIT_SCORE'] = calculate_market_fit_score(
                    st.session_state.df, st.session_state.weights)
                
                st.success(f"✅ Loaded {len(df):,} institutions")
                st.rerun()
            else:
                st.error("Failed to load data. Check file paths.")


def render_sidebar():
    """Render sidebar filters."""
    with st.sidebar:
        st.markdown("## 🎛️ Filters")
        
        # Quick preset
        preset = st.selectbox("Quick Preset", ["None"] + list(FILTER_PRESETS.keys()),
                              format_func=lambda x: FILTER_PRESETS[x]['name'] if x in FILTER_PRESETS else "Custom")
        if preset != "None" and st.button("Apply Preset"):
            st.session_state.filters = FILTER_PRESETS[preset]['filters'].copy()
            st.rerun()
        
        st.markdown("---")
        
        # Degree lens
        degree_lens = st.radio("🎓 Degree Lens", ["Both", "Graduate", "Undergraduate"], horizontal=True)
        st.session_state.filters['degree_lens'] = degree_lens
        
        # Geography (if available)
        if st.session_state.has_geography:
            st.markdown("### 📍 Geography")
            regions = st.multiselect("Regions", options=REGION_LIST)
            st.session_state.filters['regions'] = regions
            
            if 'STABBR' in st.session_state.df.columns:
                all_states = sorted(st.session_state.df['STABBR'].dropna().unique())
                states = st.multiselect("States", options=all_states)
                st.session_state.filters['states'] = states
        else:
            st.markdown('<div class="callout-warning">📍 Add HD file to unlock geography filters</div>',
                        unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Institution type
        st.markdown("### 🏫 Institution Type")
        control_opts = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}
        control = st.multiselect("Control", options=list(control_opts.keys()),
                                 format_func=lambda x: control_opts[x])
        st.session_state.filters['control'] = control
        
        level_opts = {1: "4-year+", 2: "2-year", 3: "<2-year"}
        iclevel = st.multiselect("Level", options=list(level_opts.keys()),
                                 format_func=lambda x: level_opts[x])
        st.session_state.filters['iclevel'] = iclevel
        
        # Award levels
        st.markdown("### 📜 Award Levels")
        award_levels = st.multiselect("Offerings",
                                      options=["Associates", "Bachelors", "Masters", "Doctoral", "Certificates"])
        st.session_state.filters['award_levels'] = award_levels
        
        # Distance ed
        st.markdown("### 💻 Distance Ed")
        distance_ed = st.multiselect("Modality", options=["Exclusive DE", "Some DE", "No DE"])
        st.session_state.filters['distance_ed'] = distance_ed
        
        # Enrollment
        st.markdown("### 👥 Enrollment")
        min_enroll = st.number_input("Min", min_value=0, value=0, step=100)
        max_enroll = st.number_input("Max (0=none)", min_value=0, value=0, step=1000)
        st.session_state.filters['min_enrollment'] = min_enroll if min_enroll > 0 else None
        st.session_state.filters['max_enrollment'] = max_enroll if max_enroll > 0 else None
        
        # Data quality
        hide_low = st.checkbox("Hide low-quality records")
        st.session_state.filters['hide_low_quality'] = hide_low
        
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state.filters = {}
            st.rerun()


def render_market_overview(df: pd.DataFrame):
    """Render Market Overview tab."""
    st.markdown("## 📊 Market Overview")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Institutions", f"{len(df):,}")
    with col2:
        total_enroll = df['ENRTOT'].sum() if 'ENRTOT' in df.columns else 0
        st.metric("Total Enrollment", f"{total_enroll:,.0f}")
    with col3:
        grad_enroll = df['EFGRAD'].sum() if 'EFGRAD' in df.columns else 0
        st.metric("Grad Enrollment", f"{grad_enroll:,.0f}")
    with col4:
        if 'PCTDEEXC' in df.columns:
            avg_de = df['PCTDEEXC'].mean()
            st.metric("Avg % Online-Only", f"{avg_de:.1f}%")
    with col5:
        if 'MARKET_FIT_SCORE' in df.columns:
            avg_score = df['MARKET_FIT_SCORE'].mean()
            st.metric("Avg Market Fit", f"{avg_score:.1f}")
    
    st.markdown("---")
    
    # Key takeaways
    st.markdown("### 💡 Key Takeaways")
    takeaways = generate_key_takeaways(df, st.session_state.filters)
    for t in takeaways:
        st.markdown(f"- {t}")
    
    st.markdown("---")
    
    # Charts
    show_more = st.checkbox("Show additional metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sector_breakdown(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_de_breakdown(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Map (if geography available)
    if st.session_state.has_geography:
        st.markdown("### 🗺️ Geographic Distribution")
        state_summary = get_state_summary(df)
        if not state_summary.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = create_state_choropleth(state_summary, 'Institution_Count', 'Institutions by State')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(state_summary.head(15), hide_index=True, use_container_width=True)
        
        # Point map
        fig = create_institution_map(df, 'MARKET_FIT_SCORE')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    if show_more:
        st.markdown("### Additional Metrics")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_enrollment_by_control(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_tuition_boxplot(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        fig = create_mobility_scatter(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_institution_explorer(df: pd.DataFrame):
    """Render Institution Explorer tab."""
    st.markdown("## 🔍 Institution Explorer")
    
    # Search
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        if 'INSTNM' in df.columns:
            search = st.text_input("🔎 Search institutions", placeholder="Enter institution name...")
            if search:
                df = df[df['INSTNM'].str.contains(search, case=False, na=False)]
    with search_col2:
        unitid_search = st.text_input("UNITID", placeholder="e.g., 100654")
        if unitid_search:
            try:
                df = df[df['UNITID'] == int(unitid_search)]
            except:
                pass
    
    # Column selector
    all_cols = df.columns.tolist()
    default_cols = [c for c in ['INSTNM', 'CITY', 'STABBR', 'MARKET_FIT_SCORE', 'ENRTOT', 
                                'EFGRAD', 'GRAD_FOCUS_INDEX', 'ONLINE_INTENSITY', 'MOBILITY_INDEX',
                                'PCTDEEXC', 'HAS_MASTERS', 'HAS_DOCTORAL', 'TOTAL_COST_IN_STATE',
                                'DATA_QUALITY'] if c in all_cols]
    
    with st.expander("⚙️ Customize columns"):
        selected_cols = st.multiselect("Columns to display", options=all_cols, default=default_cols)
    
    if not selected_cols:
        selected_cols = default_cols
    
    # Sort
    sort_col = st.selectbox("Sort by", options=selected_cols, 
                            index=selected_cols.index('MARKET_FIT_SCORE') if 'MARKET_FIT_SCORE' in selected_cols else 0)
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    
    display_df = df[selected_cols].sort_values(sort_col, ascending=(sort_order == "Ascending"))
    
    st.markdown(f"**Showing {len(display_df):,} institutions**")
    
    # Table
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=500
    )
    
    # Export
    csv = display_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "institutions.csv", "text/csv", use_container_width=True)
    
    # Institution detail
    if 'INSTNM' in df.columns:
        st.markdown("---")
        st.markdown("### 📋 Institution Detail")
        inst_list = df['INSTNM'].dropna().tolist()
        selected_inst = st.selectbox("Select institution", options=[""] + inst_list)
        
        if selected_inst:
            inst_row = df[df['INSTNM'] == selected_inst].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{selected_inst}**")
                if 'CITY' in inst_row and 'STABBR' in inst_row:
                    st.markdown(f"📍 {inst_row['CITY']}, {inst_row['STABBR']}")
                if 'ENRTOT' in inst_row:
                    st.metric("Total Enrollment", f"{inst_row['ENRTOT']:,.0f}")
            with col2:
                if 'MARKET_FIT_SCORE' in inst_row:
                    st.metric("Market Fit Score", f"{inst_row['MARKET_FIT_SCORE']:.1f}")
                if 'GRAD_FOCUS_INDEX' in inst_row:
                    st.metric("Grad Focus Index", f"{inst_row['GRAD_FOCUS_INDEX']:.1f}")
            with col3:
                if 'ONLINE_INTENSITY' in inst_row:
                    st.metric("Online Intensity", f"{inst_row['ONLINE_INTENSITY']:.1f}")
                if 'MOBILITY_INDEX' in inst_row:
                    st.metric("Mobility Index", f"{inst_row['MOBILITY_INDEX']:.1f}%")
            
            # Radar chart
            fig = create_market_fit_components(inst_row, st.session_state.weights)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Targeting implications
            st.markdown("**🎯 Targeting Implications:**")
            implications = []
            if inst_row.get('ONLINE_INTENSITY', 0) > 50:
                implications.append("High online share → broader geographic targeting; radius-free digital campaigns")
            if inst_row.get('MOBILITY_INDEX', 0) > 40:
                implications.append("High out-of-state draw → national audience potential; LAL audiences from current markets")
            if inst_row.get('GRAD_FOCUS_INDEX', 0) > 60:
                implications.append("Graduate focus → LinkedIn emphasis; professional/career messaging")
            if inst_row.get('PCTDEEXC', 0) < 10 and inst_row.get('PCTDENON', 0) > 80:
                implications.append("Campus-focused → local/regional geo-targeting; commuter radius strategy")
            for imp in implications:
                st.markdown(f"- {imp}")


def render_strategy_builder(df: pd.DataFrame):
    """Render Strategy Builder tab."""
    st.markdown("## 🎯 Strategy Builder")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Campaign Parameters")
        
        program_level = st.radio("Program Level", ["Graduate", "Undergraduate", "Both"])
        modality = st.radio("Modality", ["Online/Hybrid", "On-Campus", "All"])
        
        st.markdown("### Target Persona")
        persona = st.selectbox("Persona", [
            "Adult Learner (25+, working)",
            "Early Career (recent grad, career pivot)",
            "Traditional Student (18-22)",
            "Executive/Professional",
            "General"
        ])
        
        st.markdown("### Geographic Approach")
        geo_approach = st.selectbox("Approach", [
            "National (all states)",
            "Regional cluster",
            "State cluster (top 10)",
            "Local radius"
        ])
        
        st.markdown("### Market Fit Weights")
        with st.expander("Customize scoring weights"):
            preset_weights = st.selectbox("Weight preset", list(WEIGHT_PRESETS.keys()),
                                          format_func=lambda x: WEIGHT_PRESETS[x]['name'])
            if st.button("Apply weight preset"):
                st.session_state.weights = WEIGHT_PRESETS[preset_weights]['weights'].copy()
                st.rerun()
            
            st.markdown("**Manual adjustment:**")
            for key, default in st.session_state.weights.items():
                st.session_state.weights[key] = st.slider(
                    key.replace('_', ' ').title(),
                    0.0, 1.0, default, 0.05
                )
        
        if st.button("🔄 Recalculate Scores", type="primary"):
            st.session_state.df['MARKET_FIT_SCORE'] = calculate_market_fit_score(
                st.session_state.df, st.session_state.weights)
            st.success("Scores updated!")
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Recommended Markets")
        
        # Apply strategy filters to df
        strategy_df = df.copy()
        
        if program_level == "Graduate":
            if 'HAS_MASTERS' in strategy_df.columns:
                strategy_df = strategy_df[(strategy_df['HAS_MASTERS'] == 1) | (strategy_df['HAS_DOCTORAL'] == 1)]
        elif program_level == "Undergraduate":
            if 'HAS_BACHELORS' in strategy_df.columns:
                strategy_df = strategy_df[(strategy_df['HAS_ASSOCIATES'] == 1) | (strategy_df['HAS_BACHELORS'] == 1)]
        
        if modality == "Online/Hybrid" and 'PCTDEEXC' in strategy_df.columns:
            strategy_df = strategy_df[(strategy_df['PCTDEEXC'] > 20) | (strategy_df['PCTDESOM'] > 30)]
        elif modality == "On-Campus" and 'PCTDENON' in strategy_df.columns:
            strategy_df = strategy_df[strategy_df['PCTDENON'] > 50]
        
        # Recalculate with current weights
        strategy_df['MARKET_FIT_SCORE'] = calculate_market_fit_score(strategy_df, st.session_state.weights)
        
        # State summary
        if st.session_state.has_geography:
            state_summary = get_state_summary(strategy_df)
            
            if not state_summary.empty and 'MARKET_FIT_SCORE' in state_summary.columns:
                top_states = state_summary.nlargest(15, 'MARKET_FIT_SCORE')
                
                fig = create_state_ranking_bar(top_states, 'MARKET_FIT_SCORE', 15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Top Recommended States:**")
                st.dataframe(top_states[['State', 'Institution_Count', 'Total_Enrollment', 'MARKET_FIT_SCORE']],
                             hide_index=True, use_container_width=True)
        
        # Rationale
        st.markdown("### 📝 Strategy Rationale")
        
        rationale_bullets = []
        
        if len(strategy_df) > 0:
            avg_online = strategy_df['PCTDEEXC'].mean() if 'PCTDEEXC' in strategy_df.columns else 0
            avg_mobility = strategy_df['MOBILITY_INDEX'].mean() if 'MOBILITY_INDEX' in strategy_df.columns else 0
            avg_grad = strategy_df['GRAD_PCT'].mean() if 'GRAD_PCT' in strategy_df.columns else 0
            
            rationale_bullets.append(f"Analyzed {len(strategy_df):,} matching institutions")
            
            if avg_online > 30:
                rationale_bullets.append(f"High online presence ({avg_online:.0f}% avg exclusive DE) supports national digital targeting")
            if avg_mobility > 30:
                rationale_bullets.append(f"Strong mobility index ({avg_mobility:.0f}%) indicates national/international student draw")
            if avg_grad > 25:
                rationale_bullets.append(f"Graduate-heavy market ({avg_grad:.0f}% grad enrollment) favors LinkedIn & professional platforms")
            
            if program_level == "Graduate" and modality == "Online/Hybrid":
                rationale_bullets.append("Online graduate focus: expand beyond local radii; emphasize career outcomes messaging")
            elif program_level == "Undergraduate" and modality == "On-Campus":
                rationale_bullets.append("Traditional undergraduate: focus on regional commuter radius; campus experience messaging")
            
            if persona == "Adult Learner (25+, working)":
                rationale_bullets.append("Adult learner persona: emphasize flexibility, career advancement, ROI")
            elif persona == "Executive/Professional":
                rationale_bullets.append("Executive persona: premium positioning; LinkedIn sponsored content; industry-specific targeting")
        
        for bullet in rationale_bullets:
            st.markdown(f"• {bullet}")
        
        # Platform tips
        st.markdown("### 💡 Platform Targeting Tips")
        st.markdown("""
        - **Meta/Facebook**: Interest + behavioral targeting; LAL audiences from current inquiries
        - **LinkedIn**: Job title + seniority for grad/professional; industry-specific for specialized programs
        - **Google Ads**: Geo-modify bids for top-fit states; program-specific keywords
        - **Reddit**: Subreddits by field of study; career advice communities for adult learners
        """)
        
        # Exports
        st.markdown("---")
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.session_state.has_geography and 'state_summary' in dir():
                csv = state_summary.to_csv(index=False)
                st.download_button("📥 Download Market Rankings", csv, "market_rankings.csv", "text/csv")
        
        with col_exp2:
            summary_text = format_strategy_summary(strategy_df, 
                                                    state_summary if st.session_state.has_geography and 'state_summary' in dir() else pd.DataFrame(),
                                                    st.session_state.filters,
                                                    st.session_state.weights)
            st.download_button("📄 Download Strategy Summary", summary_text, "strategy_summary.txt", "text/plain")


def main():
    """Main application entry point."""
    
    if not st.session_state.data_loaded:
        render_data_loader()
        return
    
    # Sidebar
    render_sidebar()
    
    # Apply filters
    df = apply_filters(st.session_state.df, st.session_state.filters)
    
    # Header
    st.markdown('<p class="main-header">🎯 IPEDS Market Targeting Tool</p>', unsafe_allow_html=True)
    
    # Status bar
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        geo_status = "✅ Geography enabled" if st.session_state.has_geography else "⚠️ Geography disabled (add HD file)"
        st.markdown(f"**Status:** {geo_status}")
    with col2:
        st.markdown(f"**Filtered:** {len(df):,} of {len(st.session_state.df):,} institutions")
    with col3:
        if st.button("🔄 Reload Data"):
            st.session_state.data_loaded = False
            st.rerun()
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🔍 Institution Explorer", "🎯 Strategy Builder"])
    
    with tab1:
        render_market_overview(df)
    
    with tab2:
        render_institution_explorer(df)
    
    with tab3:
        render_strategy_builder(df)


if __name__ == "__main__":
    main()
