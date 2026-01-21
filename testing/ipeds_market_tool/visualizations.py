"""
Visualization module for IPEDS Market Targeting Tool.
Contains functions for creating charts, maps, and visual components.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List
import streamlit as st


# Color palette - professional, modern
COLORS = {
    'primary': '#2E4057',      # Dark blue
    'secondary': '#048A81',    # Teal
    'accent': '#54C6EB',       # Light blue
    'warning': '#F4A024',      # Orange
    'success': '#8AC926',      # Green
    'danger': '#FF595E',       # Red
    'neutral': '#6C757D',      # Gray
    'background': '#F8F9FA',   # Light gray
    'text': '#212529',         # Dark gray
}

# Categorical color sequence
COLOR_SEQUENCE = [
    '#2E4057', '#048A81', '#54C6EB', '#F4A024', 
    '#8AC926', '#FF595E', '#9B5DE5', '#00BBF9'
]


def create_kpi_card(value, label: str, delta: Optional[float] = None, 
                    prefix: str = "", suffix: str = "", 
                    format_type: str = "number") -> None:
    """Create a styled KPI metric card using Streamlit."""
    
    if format_type == "number":
        if isinstance(value, (int, float)):
            if value >= 1_000_000:
                display_value = f"{prefix}{value/1_000_000:.1f}M{suffix}"
            elif value >= 1_000:
                display_value = f"{prefix}{value/1_000:.1f}K{suffix}"
            else:
                display_value = f"{prefix}{value:,.0f}{suffix}"
        else:
            display_value = str(value)
    elif format_type == "percent":
        display_value = f"{prefix}{value:.1f}%{suffix}"
    elif format_type == "currency":
        display_value = f"${value:,.0f}"
    else:
        display_value = str(value)
    
    if delta is not None:
        st.metric(label=label, value=display_value, delta=f"{delta:+.1f}%")
    else:
        st.metric(label=label, value=display_value)


def create_enrollment_histogram(df: pd.DataFrame, column: str = 'ENRTOT', 
                                 title: str = "Enrollment Distribution") -> go.Figure:
    """Create histogram of enrollment distribution."""
    fig = px.histogram(
        df[df[column] > 0],
        x=column,
        nbins=50,
        title=title,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        xaxis_title="Enrollment",
        yaxis_title="Number of Institutions",
        showlegend=False,
        template='plotly_white',
        height=350
    )
    
    return fig


def create_sector_breakdown(df: pd.DataFrame) -> go.Figure:
    """Create donut chart of sector breakdown."""
    if 'SECTOR_LABEL' not in df.columns:
        return None
    
    sector_counts = df['SECTOR_LABEL'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']
    
    fig = px.pie(
        sector_counts,
        values='Count',
        names='Sector',
        title='Institution Sector Breakdown',
        hole=0.4,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    
    fig.update_layout(
        template='plotly_white',
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )
    
    return fig


def create_de_breakdown(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of distance education breakdown."""
    de_cols = ['PCTDEEXC', 'PCTDESOM', 'PCTDENON']
    if not all(c in df.columns for c in de_cols):
        return None
    
    # Calculate averages
    de_data = {
        'Category': ['Exclusive DE', 'Some DE', 'No DE'],
        'Percentage': [
            df['PCTDEEXC'].mean(),
            df['PCTDESOM'].mean(),
            df['PCTDENON'].mean()
        ]
    }
    
    fig = px.bar(
        pd.DataFrame(de_data),
        x='Category',
        y='Percentage',
        title='Average Distance Education Mix',
        color='Category',
        color_discrete_map={
            'Exclusive DE': COLORS['primary'],
            'Some DE': COLORS['secondary'],
            'No DE': COLORS['neutral']
        }
    )
    
    fig.update_layout(
        yaxis_title='Average Percentage',
        showlegend=False,
        template='plotly_white',
        height=350
    )
    
    return fig


def create_grad_ug_comparison(df: pd.DataFrame) -> go.Figure:
    """Create comparison of grad vs undergrad enrollment."""
    if 'EFUG' not in df.columns or 'EFGRAD' not in df.columns:
        return None
    
    total_ug = df['EFUG'].sum()
    total_grad = df['EFGRAD'].sum()
    
    fig = go.Figure(data=[
        go.Bar(name='Undergraduate', x=['Enrollment'], y=[total_ug], 
               marker_color=COLORS['primary']),
        go.Bar(name='Graduate', x=['Enrollment'], y=[total_grad], 
               marker_color=COLORS['secondary'])
    ])
    
    fig.update_layout(
        title='Total Enrollment by Level',
        barmode='group',
        template='plotly_white',
        height=350,
        yaxis_title='Total Students'
    )
    
    return fig


def create_mobility_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of online intensity vs mobility."""
    if 'ONLINE_INTENSITY' not in df.columns or 'MOBILITY_INDEX' not in df.columns:
        return None
    
    # Sample if too many points
    plot_df = df.copy()
    if len(plot_df) > 1000:
        plot_df = plot_df.sample(1000, random_state=42)
    
    hover_cols = ['INSTNM'] if 'INSTNM' in plot_df.columns else []
    
    fig = px.scatter(
        plot_df,
        x='ONLINE_INTENSITY',
        y='MOBILITY_INDEX',
        size='FTE' if 'FTE' in plot_df.columns else None,
        color='GRAD_FOCUS_INDEX' if 'GRAD_FOCUS_INDEX' in plot_df.columns else None,
        hover_name='INSTNM' if 'INSTNM' in plot_df.columns else None,
        title='Online Intensity vs Student Mobility',
        color_continuous_scale='Viridis',
        opacity=0.6
    )
    
    fig.update_layout(
        xaxis_title='Online Intensity (0-100)',
        yaxis_title='Mobility Index (Out-of-State + Foreign %)',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_state_choropleth(df: pd.DataFrame, value_column: str = 'Institution_Count',
                            title: str = "Institutions by State") -> go.Figure:
    """Create US state choropleth map."""
    if 'State' not in df.columns:
        return None
    
    fig = px.choropleth(
        df,
        locations='State',
        locationmode='USA-states',
        color=value_column,
        scope='usa',
        title=title,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        template='plotly_white',
        height=450,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_institution_map(df: pd.DataFrame, color_by: str = 'MARKET_FIT_SCORE') -> go.Figure:
    """Create scatter map of institutions."""
    if 'LATITUDE' not in df.columns or 'LONGITUD' not in df.columns:
        return None
    
    # Filter to valid coordinates
    plot_df = df[
        (df['LATITUDE'].notna()) & 
        (df['LONGITUD'].notna()) &
        (df['LATITUDE'].between(24, 50)) &
        (df['LONGITUD'].between(-125, -66))
    ].copy()
    
    if plot_df.empty:
        return None
    
    # Sample if too many
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000, random_state=42)
    
    fig = px.scatter_geo(
        plot_df,
        lat='LATITUDE',
        lon='LONGITUD',
        hover_name='INSTNM' if 'INSTNM' in plot_df.columns else None,
        color=color_by if color_by in plot_df.columns else None,
        size='FTE' if 'FTE' in plot_df.columns else None,
        size_max=20,
        scope='usa',
        title='Institution Locations',
        color_continuous_scale='Viridis',
        opacity=0.7
    )
    
    fig.update_layout(
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_market_fit_components(row: pd.Series, weights: Dict[str, float]) -> go.Figure:
    """Create radar chart showing Market Fit Score components for a single institution."""
    categories = ['Grad Focus', 'Online Intensity', 'Mobility', 'Enrollment Scale', 'Cost Access.']
    
    # Map to column names
    col_map = {
        'Grad Focus': 'GRAD_FOCUS_INDEX',
        'Online Intensity': 'ONLINE_INTENSITY',
        'Mobility': 'MOBILITY_INDEX',
        'Enrollment Scale': 'ENROLLMENT_SCALE',
        'Cost Access.': 'COST_ACCESSIBILITY'
    }
    
    values = [row.get(col_map[cat], 0) for cat in categories]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=f'rgba(46, 64, 87, 0.3)',
        line_color=COLORS['primary']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title='Market Fit Components',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_state_ranking_bar(state_summary: pd.DataFrame, 
                             metric: str = 'MARKET_FIT_SCORE',
                             top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of top states by metric."""
    if metric not in state_summary.columns:
        metric = 'Institution_Count'
    
    sorted_df = state_summary.nlargest(top_n, metric)
    
    fig = px.bar(
        sorted_df,
        x=metric,
        y='State',
        orientation='h',
        title=f'Top {top_n} States by {metric.replace("_", " ")}',
        color=metric,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_tuition_boxplot(df: pd.DataFrame) -> go.Figure:
    """Create boxplot comparing in-state vs out-of-state tuition."""
    if 'TOTAL_COST_IN_STATE' not in df.columns or 'TOTAL_COST_OUT_STATE' not in df.columns:
        return None
    
    # Prepare data
    in_state = df['TOTAL_COST_IN_STATE'].dropna()
    out_state = df['TOTAL_COST_OUT_STATE'].dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=in_state[in_state > 0],
        name='In-State',
        marker_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Box(
        y=out_state[out_state > 0],
        name='Out-of-State',
        marker_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        title='Tuition + Fees Distribution',
        yaxis_title='Annual Cost ($)',
        template='plotly_white',
        height=350,
        showlegend=True
    )
    
    return fig


def create_enrollment_by_control(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart of enrollment by control type."""
    if 'CONTROL_LABEL' not in df.columns:
        return None
    
    grouped = df.groupby('CONTROL_LABEL').agg({
        'ENRTOT': 'sum',
        'EFUG': 'sum',
        'EFGRAD': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Undergraduate',
        x=grouped['CONTROL_LABEL'],
        y=grouped['EFUG'],
        marker_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Bar(
        name='Graduate',
        x=grouped['CONTROL_LABEL'],
        y=grouped['EFGRAD'],
        marker_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        title='Enrollment by Institution Control',
        barmode='group',
        template='plotly_white',
        height=350,
        xaxis_title='Control Type',
        yaxis_title='Total Enrollment'
    )
    
    return fig


def create_score_distribution(df: pd.DataFrame, score_col: str = 'MARKET_FIT_SCORE') -> go.Figure:
    """Create histogram of Market Fit Score distribution with density."""
    if score_col not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x=score_col,
        nbins=30,
        title='Market Fit Score Distribution',
        color_discrete_sequence=[COLORS['primary']],
        marginal='box'
    )
    
    fig.update_layout(
        xaxis_title='Market Fit Score',
        yaxis_title='Count',
        template='plotly_white',
        height=350
    )
    
    return fig


def style_dataframe(df: pd.DataFrame, highlight_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply styling to dataframe for display."""
    # Return subset of columns for display
    display_cols = []
    
    # Priority order for columns
    priority_cols = [
        'INSTNM', 'CITY', 'STABBR', 'MARKET_FIT_SCORE', 'ENRTOT', 'FTE',
        'GRAD_FOCUS_INDEX', 'ONLINE_INTENSITY', 'MOBILITY_INDEX',
        'PCTDEEXC', 'PCTDESOM', 'GRAD_PCT', 'HAS_MASTERS', 'HAS_DOCTORAL',
        'TOTAL_COST_IN_STATE', 'DATA_QUALITY'
    ]
    
    for col in priority_cols:
        if col in df.columns:
            display_cols.append(col)
    
    return df[display_cols] if display_cols else df


def generate_key_takeaways(df: pd.DataFrame, filters: Dict) -> List[str]:
    """Generate auto-generated key takeaways based on current data."""
    takeaways = []
    
    n_institutions = len(df)
    if n_institutions < 10:
        takeaways.append(f"⚠️ Small sample size ({n_institutions} institutions) - interpret with caution")
    else:
        takeaways.append(f"📊 Analyzing {n_institutions:,} institutions")
    
    # Enrollment stats
    if 'ENRTOT' in df.columns:
        total_enroll = df['ENRTOT'].sum()
        takeaways.append(f"👥 Total enrollment: {total_enroll:,.0f} students")
    
    # Graduate focus
    if 'GRAD_PCT' in df.columns:
        avg_grad_pct = df['GRAD_PCT'].mean()
        if avg_grad_pct > 30:
            takeaways.append(f"🎓 Strong graduate focus: {avg_grad_pct:.1f}% average graduate enrollment")
        elif avg_grad_pct < 10:
            takeaways.append(f"📚 Predominantly undergraduate: only {avg_grad_pct:.1f}% graduate enrollment")
    
    # Online intensity
    if 'PCTDEEXC' in df.columns:
        avg_online = df['PCTDEEXC'].mean()
        if avg_online > 30:
            takeaways.append(f"💻 High online presence: {avg_online:.1f}% students exclusively online - suggests broader geographic targeting")
        elif avg_online < 5:
            takeaways.append(f"🏫 Primarily campus-based: only {avg_online:.1f}% exclusively online")
    
    # Mobility
    if 'MOBILITY_INDEX' in df.columns:
        avg_mobility = df['MOBILITY_INDEX'].mean()
        if avg_mobility > 40:
            takeaways.append(f"✈️ High student mobility: {avg_mobility:.1f}% from out-of-state/international - indicates national/global draw")
        elif avg_mobility < 15:
            takeaways.append(f"🏠 Regional focus: {avg_mobility:.1f}% out-of-state students - consider local market targeting")
    
    # Market Fit Score
    if 'MARKET_FIT_SCORE' in df.columns:
        avg_score = df['MARKET_FIT_SCORE'].mean()
        high_fit = (df['MARKET_FIT_SCORE'] > 70).sum()
        if high_fit > 0:
            takeaways.append(f"🎯 {high_fit} institutions with high market fit (>70) for selected criteria")
    
    # Cost
    if 'TOTAL_COST_IN_STATE' in df.columns:
        median_cost = df['TOTAL_COST_IN_STATE'].median()
        if median_cost > 0:
            takeaways.append(f"💰 Median in-state cost: ${median_cost:,.0f}")
    
    return takeaways


def format_strategy_summary(df: pd.DataFrame, state_summary: pd.DataFrame,
                            filters: Dict, weights: Dict) -> str:
    """Generate formatted strategy summary text for export."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("IPEDS MARKET TARGETING STRATEGY SUMMARY")
    summary.append("=" * 60)
    summary.append("")
    
    # Filter summary
    summary.append("APPLIED FILTERS:")
    summary.append("-" * 40)
    for key, value in filters.items():
        if value:
            summary.append(f"  {key}: {value}")
    summary.append("")
    
    # Market overview
    summary.append("MARKET OVERVIEW:")
    summary.append("-" * 40)
    summary.append(f"  Total institutions: {len(df):,}")
    if 'ENRTOT' in df.columns:
        summary.append(f"  Total enrollment: {df['ENRTOT'].sum():,.0f}")
    if 'FTE' in df.columns:
        summary.append(f"  Total FTE: {df['FTE'].sum():,.0f}")
    if 'EFGRAD' in df.columns:
        summary.append(f"  Graduate enrollment: {df['EFGRAD'].sum():,.0f}")
    summary.append("")
    
    # Weight configuration
    summary.append("SCORING WEIGHTS:")
    summary.append("-" * 40)
    for key, value in weights.items():
        summary.append(f"  {key}: {value*100:.0f}%")
    summary.append("")
    
    # Top states
    if not state_summary.empty and 'MARKET_FIT_SCORE' in state_summary.columns:
        summary.append("TOP 10 RECOMMENDED STATES:")
        summary.append("-" * 40)
        top_states = state_summary.nlargest(10, 'MARKET_FIT_SCORE')
        for _, row in top_states.iterrows():
            summary.append(
                f"  {row['State']}: Score {row['MARKET_FIT_SCORE']:.1f}, "
                f"{row['Institution_Count']} institutions, "
                f"{row['Total_Enrollment']:,.0f} students"
            )
        summary.append("")
    
    # Key takeaways
    summary.append("KEY TAKEAWAYS:")
    summary.append("-" * 40)
    takeaways = generate_key_takeaways(df, filters)
    for t in takeaways:
        summary.append(f"  {t}")
    summary.append("")
    
    # Targeting recommendations
    summary.append("TARGETING RECOMMENDATIONS:")
    summary.append("-" * 40)
    
    if 'PCTDEEXC' in df.columns and df['PCTDEEXC'].mean() > 30:
        summary.append("  • High online share suggests national/broad geographic targeting")
        summary.append("  • Consider radius-free targeting for digital campaigns")
    
    if 'MOBILITY_INDEX' in df.columns and df['MOBILITY_INDEX'].mean() > 30:
        summary.append("  • Strong out-of-state draw supports national campaigns")
        summary.append("  • Look-alike audiences based on current student origins may perform well")
    
    if 'GRAD_PCT' in df.columns and df['GRAD_PCT'].mean() > 25:
        summary.append("  • Graduate focus suggests professional platform emphasis (LinkedIn)")
        summary.append("  • Target career-focused messaging and professional development angles")
    
    summary.append("")
    summary.append("=" * 60)
    summary.append("Generated by IPEDS Market Targeting Tool")
    summary.append("=" * 60)
    
    return "\n".join(summary)
