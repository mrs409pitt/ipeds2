# IPEDS Market Targeting Tool

A decision-support application for higher education paid media managers and marketing directors to determine **where paid media should run geographically** for graduate and undergraduate program campaigns.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This tool analyzes IPEDS (Integrated Postsecondary Education Data System) data to help higher-ed marketers:

- **Identify high-potential geographic markets** based on institutional characteristics
- **Compare institutions** by online intensity, graduate focus, student mobility, and cost
- **Generate targeting recommendations** for paid media campaigns across Meta, LinkedIn, Google, and Reddit
- **Export actionable data** for platform targeting and strategy documentation

### Who It's For

- **Paid Media Managers**: Need market selection, targeting rationale, and exportable institution/state lists
- **Marketing Directors**: Need high-level rollups and "why this geography" summaries for strategy docs

### The 60-Second Answer

> "Given an online/hybrid graduate program, which states/metros should we prioritize and why—based on IPEDS signals?"

Load the tool → Set Graduate + Online filters → View Strategy Builder → Get your ranked state recommendations with transparent scoring.

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone or download the tool
cd ipeds_market_tool

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create `requirements.txt`:

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
pyarrow>=14.0.0
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Place these IPEDS CSV files in a data directory (e.g., `/data/ipeds/`):

| File | Description | Required |
|------|-------------|----------|
| `drvef2024.csv` | Enrollment totals, demographics, residence mix | ✅ Yes |
| `drvc2024.csv` | Degree/certificate offerings by award level | ✅ Yes |
| `ef2024a_dist.csv` | Distance education enrollment | ✅ Yes |
| `cost1_2024.csv` | Tuition, fees, room/board | ✅ Yes |
| `flags2024.csv` | Data quality/imputation flags | ✅ Yes |
| `hd2024.csv` | Institution directory (name, city, state, lat/long) | 📍 Optional |

> **Note:** The HD (header/directory) file enables geographic mapping and state/city filters. Without it, the tool runs in "demo mode" with UNITID-only analysis.

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Configure Data Path

1. Enter your data directory path (e.g., `/data/ipeds/`)
2. Select HD file source (directory, upload, or skip)
3. Click "Load Data"

---

## 📊 Key Features

### Market Overview Tab

- **KPI Cards**: Institution count, total enrollment, graduate enrollment, online percentages
- **Sector Breakdown**: Donut chart of public/private/for-profit mix
- **Distance Education Mix**: Average exclusive vs. some vs. no distance ed
- **Geographic Map** (requires HD file): Choropleth by state + point map of institutions
- **Key Takeaways**: Auto-generated insights based on filtered data

### Institution Explorer Tab

- **Searchable Table**: Filter by name or UNITID
- **Customizable Columns**: Select which metrics to display
- **Sort & Export**: Download filtered institution lists as CSV
- **Institution Detail**: Select any institution for a detailed profile with radar chart

### Strategy Builder Tab

- **Campaign Parameters**: Set program level, modality, and target persona
- **Weight Customization**: Adjust Market Fit Score component weights
- **State Rankings**: See top markets with transparent scores
- **Strategy Rationale**: Auto-generated bullets explaining recommendations
- **Platform Tips**: Operational guidance for Meta, LinkedIn, Google, Reddit
- **Export Options**: Download market rankings CSV or strategy summary text

---

## 🔢 Key Metrics & Scoring

### Derived Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Grad Focus Index** (0-100) | Strength of graduate offerings | 25 pts (has masters) + 25 pts (has doctoral) + grad % capped at 50 |
| **Online Intensity** (0-100) | Distance education saturation | (% exclusive DE × 1.0) + (% some DE × 0.5) |
| **Mobility Index** (%) | Out-of-state + international draw | Out-of-state % + Foreign % from first-time students |
| **Enrollment Scale** (0-100) | Normalized FTE size | Decile ranking of FTE |
| **Cost Accessibility** (0-100) | Inverse of in-state cost | 100 − percentile rank of tuition+fees |

### Market Fit Score

A weighted composite (0-100) with customizable weights. Default configuration favors graduate + online:

| Component | Default Weight |
|-----------|----------------|
| Grad Focus | 30% |
| Online Intensity | 25% |
| Mobility Index | 20% |
| Enrollment Scale | 15% |
| Cost Accessibility | 10% |

**How to interpret**: Higher scores indicate better alignment with the selected targeting criteria. A score of 70+ suggests strong market fit.

### Weight Presets

- **Graduate Online Focus** (default): Prioritizes grad programs + distance ed
- **Undergraduate Regional**: Emphasizes UG enrollment + local presence
- **Online Market Focus**: Maximizes online intensity weight
- **National Reach**: Prioritizes mobility/geographic draw

---

## 🗂️ Data Files Reference

### drvef2024.csv (Enrollment)
Key columns: `ENRTOT`, `FTE`, `ENRFT`, `ENRPT`, `EFUG`, `EFGRAD`, `RMOUSTTP`, `RMFRGNCP`, `PCTDEEXC`, `PCTDESOM`, `PCTDENON`

### drvc2024.csv (Degrees)
Key columns: `ASCDEG`, `BASDEG`, `MASDEG`, `DOCDEGRS`, `DOCDEGPP`, `CERT1`, `PBACERT`, `PMACERT`

### ef2024a_dist.csv (Distance Ed)
Key columns: `EFDELEV` (level code), `EFDETOT`, `EFDEEXC`, `EFDESOM`, `EFDENON`

### cost1_2024.csv (Cost)
Key columns: `TUITION2`, `TUITION3`, `FEE2`, `FEE3`, `ROOMAMT`, `BOARDAMT`

### hd2024.csv (Directory)
Key columns: `UNITID`, `INSTNM`, `CITY`, `STABBR`, `ZIP`, `LATITUDE`, `LONGITUD`, `SECTOR`, `CONTROL`, `ICLEVEL`, `LOCALE`

### flags2024.csv (Data Quality)
Key columns: `IMP_*` columns indicate imputation status

---

## 🔧 Configuration

### config.py

The configuration file contains:

- **Column Mappings**: Human-readable labels for IPEDS codes
- **Sector/Control Labels**: Lookup tables for categorical codes
- **Weight Presets**: Pre-defined scoring weight configurations
- **Filter Presets**: Quick-start filter combinations
- **Market Tags**: Pre-defined market segments (high online, grad-heavy, etc.)

### Customizing Weights

1. Go to Strategy Builder tab
2. Expand "Customize scoring weights"
3. Use sliders to adjust component weights
4. Click "Recalculate Scores" to update

### Filter Presets

Apply pre-configured filter combinations:

- **Grad Online Default**: Graduate + distance ed focus
- **UG Regional Default**: Traditional undergraduate targeting
- **Certificate Programs**: Short-term credential focus
- **Large Online Programs**: High-volume online institutions

---

## 📤 Exports

### Institution CSV
Download filtered institution list with all selected columns.

### Market Rankings CSV
State-level summary with institution count, enrollment totals, and average Market Fit Score.

### Strategy Summary (Text)
Formatted text document containing:
- Applied filters
- Market overview statistics
- Scoring weights
- Top 10 recommended states
- Key takeaways
- Targeting recommendations

---

## ⚠️ Limitations & Edge Cases

### Data Quality
- Some records have imputed values; use "Hide low-quality records" toggle
- The `flags2024.csv` file indicates imputation status

### Small Samples
- Filters that yield <10 institutions show a warning
- Aggregated metrics should be interpreted cautiously

### Geography
- Without `hd2024.csv`, maps and state filters are disabled
- The tool still functions with UNITID-only analysis

### Missing Values
- Columns with >50% missing values are renormalized in scoring
- "Unknown" is shown for missing categorical values

---

## 🏗️ Architecture

```
ipeds_market_tool/
├── app.py              # Main Streamlit application
├── data_loader.py      # Data ingestion, merging, filtering
├── visualizations.py   # Charts, maps, KPI components
├── config.py           # Column mappings, weights, presets
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Data Flow

1. **Load**: CSVs read with pandas, cached with Streamlit
2. **Merge**: Join on UNITID with validation
3. **Derive**: Calculate composite metrics
4. **Filter**: Apply user selections
5. **Score**: Calculate Market Fit Score
6. **Visualize**: Render charts and tables
7. **Export**: Generate downloadable outputs

---

## 🔒 Data Privacy

This tool processes publicly available IPEDS data. No personally identifiable information (PII) is collected or stored. All analysis happens locally in your browser session.

---

## 📝 Assumptions & Decisions

1. **Graduate focus by default**: Default weights prioritize grad + online (matches typical program marketing portfolio)
2. **Mobility as signal**: High out-of-state enrollment indicates institutions with national draw, suggesting broader targeting
3. **Online intensity scaling**: Exclusive DE counts more than "some DE" in the intensity calculation
4. **Cost as inverse metric**: Lower cost = higher accessibility score (more competitive for ads)
5. **State-level aggregation**: For strategy recommendations, we aggregate to state level (platform targeting typically works at state/DMA level)

---

## 🤝 Contributing

Suggestions and improvements welcome! Key areas for enhancement:

- [ ] Add DMA/metro-level aggregation
- [ ] Integrate IPEDS API for live data refresh
- [ ] Add time-series analysis for trend detection
- [ ] Implement saved filter configurations
- [ ] Add competitive set comparison features

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- Data source: [IPEDS](https://nces.ed.gov/ipeds/) (National Center for Education Statistics)
- Built with [Streamlit](https://streamlit.io/), [Pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/)

---

*Built for higher-ed marketers who need data-driven geographic targeting decisions.*
