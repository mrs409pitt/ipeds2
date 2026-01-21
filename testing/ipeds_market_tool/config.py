"""
Configuration file for IPEDS Market Targeting Tool
Contains column mappings, score weights, and default filter presets.
"""

# =============================================================================
# COLUMN MAPPINGS - Human-readable labels for IPEDS columns
# =============================================================================

COLUMN_LABELS = {
    # Header/Directory File (hd2024)
    'UNITID': 'Institution ID',
    'INSTNM': 'Institution Name',
    'IALIAS': 'Aliases',
    'CITY': 'City',
    'STABBR': 'State',
    'ZIP': 'ZIP Code',
    'LATITUDE': 'Latitude',
    'LONGITUD': 'Longitude',
    'SECTOR': 'Sector',
    'ICLEVEL': 'Institution Level',
    'CONTROL': 'Control (Public/Private)',
    'HLOFFER': 'Highest Level Offering',
    'UGOFFER': 'Offers Undergraduate',
    'GROFFER': 'Offers Graduate',
    'HBCU': 'HBCU Status',
    'LOCALE': 'Locale (Urbanicity)',
    'INSTSIZE': 'Institution Size Category',
    'LANDGRNT': 'Land Grant Institution',
    'OBEREG': 'Bureau of Economic Analysis Region',
    
    # Degree/Certificate Offerings (drvc2024)
    'ASCDEG': 'Associate Degrees Awarded',
    'BASDEG': 'Bachelor\'s Degrees Awarded',
    'MASDEG': 'Master\'s Degrees Awarded',
    'DOCDEGRS': 'Research Doctorates Awarded',
    'DOCDEGPP': 'Professional Doctorates Awarded',
    'DOCDEGOT': 'Other Doctorates Awarded',
    'CERT1': 'Certificates < 1 Year',
    'CERT1A': 'Certificates < 900 Hours',
    'CERT1B': 'Certificates 900-1800 Hours',
    'CERT2': 'Certificates 1-2 Years',
    'CERT4': 'Certificates 2-4 Years',
    'PBACERT': 'Post-Baccalaureate Certificates',
    'PMACERT': 'Post-Master\'s Certificates',
    
    # Enrollment (drvef2024)
    'ENRTOT': 'Total Enrollment',
    'FTE': 'Full-Time Equivalent',
    'ENRFT': 'Full-Time Enrollment',
    'ENRPT': 'Part-Time Enrollment',
    'EFUG': 'Undergraduate Enrollment',
    'EFGRAD': 'Graduate Enrollment',
    'EFUGFT': 'Undergraduate Full-Time',
    'EFUGPT': 'Undergraduate Part-Time',
    'EFGRADFT': 'Graduate Full-Time',
    'EFGRADPT': 'Graduate Part-Time',
    
    # Demographics (drvef2024)
    'PCTENRWH': '% White',
    'PCTENRBK': '% Black',
    'PCTENRHS': '% Hispanic',
    'PCTENRAP': '% Asian/Pacific Islander',
    'PCTENRAS': '% Asian',
    'PCTENRNH': '% Native Hawaiian/Pacific Islander',
    'PCTENRAN': '% American Indian/Alaska Native',
    'PCTENR2M': '% Two or More Races',
    'PCTENRUN': '% Unknown Race',
    'PCTENRNR': '% Nonresident Alien',
    'PCTENRW': '% Women',
    
    # Residence Mix (drvef2024)
    'RMINSTTN': 'In-State FT First-Time (#)',
    'RMOUSTTN': 'Out-of-State FT First-Time (#)',
    'RMFRGNCN': 'Foreign FT First-Time (#)',
    'RMUNKNWN': 'Unknown Residence (#)',
    'RMINSTTP': 'In-State FT First-Time (%)',
    'RMOUSTTP': 'Out-of-State FT First-Time (%)',
    'RMFRGNCP': 'Foreign FT First-Time (%)',
    'RMUNKNWP': 'Unknown Residence (%)',
    
    # Distance Education (drvef2024)
    'PCTDEEXC': '% Exclusive Distance Ed',
    'PCTDESOM': '% Some Distance Ed',
    'PCTDENON': '% No Distance Ed',
    
    # Distance Education Detail (ef2024a_dist)
    'EFDETOT': 'Distance Ed Total Enrollment',
    'EFDEEXC': 'Exclusively Distance Ed',
    'EFDESOM': 'Some Distance Ed',
    'EFDENON': 'No Distance Ed',
    
    # Cost (cost1_2024)
    'TUITION1': 'In-District Tuition (UG)',
    'TUITION2': 'In-State Tuition (UG)',
    'TUITION3': 'Out-of-State Tuition (UG)',
    'FEE1': 'In-District Fees (UG)',
    'FEE2': 'In-State Fees (UG)',
    'FEE3': 'Out-of-State Fees (UG)',
    'ROOMAMT': 'Room Charges',
    'BOARDAMT': 'Board Charges',
    'RMBRDAMT': 'Room & Board',
    'TUITVARY': 'Tuition Varies by Program',
}

COLUMN_TOOLTIPS = {
    'UNITID': 'Unique identifier for each institution in IPEDS',
    'SECTOR': '1=Public 4-yr, 2=Private nonprofit 4-yr, 3=Private for-profit 4-yr, 4=Public 2-yr, 5=Private nonprofit 2-yr, 6=Private for-profit 2-yr, 7-9=Less than 2-yr',
    'CONTROL': '1=Public, 2=Private nonprofit, 3=Private for-profit',
    'ICLEVEL': '1=Four or more years, 2=At least 2 but less than 4 years, 3=Less than 2 years',
    'LOCALE': '11=City Large, 12=City Midsize, 13=City Small, 21=Suburb Large, 22=Suburb Midsize, 23=Suburb Small, 31=Town Fringe, 32=Town Distant, 33=Town Remote, 41=Rural Fringe, 42=Rural Distant, 43=Rural Remote',
    'HLOFFER': 'Highest level of offering: 0=Other, 1=Postsec award<1yr, 2=Postsec award 1-2yr, 3=Associate, 4=Postsec award 2-4yr, 5=Bachelor\'s, 6=Postbacc cert, 7=Master\'s, 8=Post-master\'s cert, 9=Doctor\'s',
    'OBEREG': 'BEA Region: 1=New England, 2=Mid East, 3=Great Lakes, 4=Plains, 5=Southeast, 6=Southwest, 7=Rocky Mountains, 8=Far West, 9=Outlying Areas',
    'PCTDEEXC': 'Percentage of students enrolled exclusively in distance education courses',
    'PCTDESOM': 'Percentage of students enrolled in some but not all distance education courses',
    'RMOUSTTP': 'Percentage of first-time students from out of state - indicates geographic reach',
    'RMFRGNCP': 'Percentage of first-time students from foreign countries - indicates international reach',
    'FTE': 'Full-time equivalent enrollment: Full-time + (Part-time × 0.335737)',
}

# =============================================================================
# SECTOR AND CONTROL MAPPINGS
# =============================================================================

SECTOR_LABELS = {
    1: 'Public, 4-year or above',
    2: 'Private nonprofit, 4-year or above',
    3: 'Private for-profit, 4-year or above',
    4: 'Public, 2-year',
    5: 'Private nonprofit, 2-year',
    6: 'Private for-profit, 2-year',
    7: 'Public, less than 2-year',
    8: 'Private nonprofit, less than 2-year',
    9: 'Private for-profit, less than 2-year',
    99: 'Sector unknown (not active)',
    0: 'Administrative unit'
}

CONTROL_LABELS = {
    1: 'Public',
    2: 'Private nonprofit',
    3: 'Private for-profit'
}

ICLEVEL_LABELS = {
    1: '4-year or above',
    2: '2-year (less than 4)',
    3: 'Less than 2-year',
    -3: 'Not available'
}

LOCALE_LABELS = {
    11: 'City: Large',
    12: 'City: Midsize',
    13: 'City: Small',
    21: 'Suburb: Large',
    22: 'Suburb: Midsize',
    23: 'Suburb: Small',
    31: 'Town: Fringe',
    32: 'Town: Distant',
    33: 'Town: Remote',
    41: 'Rural: Fringe',
    42: 'Rural: Distant',
    43: 'Rural: Remote',
    -3: 'Not available'
}

BEA_REGION_LABELS = {
    0: 'US Service Schools',
    1: 'New England (CT, ME, MA, NH, RI, VT)',
    2: 'Mid East (DE, DC, MD, NJ, NY, PA)',
    3: 'Great Lakes (IL, IN, MI, OH, WI)',
    4: 'Plains (IA, KS, MN, MO, NE, ND, SD)',
    5: 'Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)',
    6: 'Southwest (AZ, NM, OK, TX)',
    7: 'Rocky Mountains (CO, ID, MT, UT, WY)',
    8: 'Far West (AK, CA, HI, NV, OR, WA)',
    9: 'Outlying Areas (AS, FM, GU, MH, MP, PR, PW, VI)'
}

# State to region mapping for geographic filters
STATE_TO_REGION = {
    'CT': 'New England', 'ME': 'New England', 'MA': 'New England', 
    'NH': 'New England', 'RI': 'New England', 'VT': 'New England',
    'DE': 'Mid East', 'DC': 'Mid East', 'MD': 'Mid East', 
    'NJ': 'Mid East', 'NY': 'Mid East', 'PA': 'Mid East',
    'IL': 'Great Lakes', 'IN': 'Great Lakes', 'MI': 'Great Lakes', 
    'OH': 'Great Lakes', 'WI': 'Great Lakes',
    'IA': 'Plains', 'KS': 'Plains', 'MN': 'Plains', 
    'MO': 'Plains', 'NE': 'Plains', 'ND': 'Plains', 'SD': 'Plains',
    'AL': 'Southeast', 'AR': 'Southeast', 'FL': 'Southeast', 
    'GA': 'Southeast', 'KY': 'Southeast', 'LA': 'Southeast', 
    'MS': 'Southeast', 'NC': 'Southeast', 'SC': 'Southeast', 
    'TN': 'Southeast', 'VA': 'Southeast', 'WV': 'Southeast',
    'AZ': 'Southwest', 'NM': 'Southwest', 'OK': 'Southwest', 'TX': 'Southwest',
    'CO': 'Rocky Mountains', 'ID': 'Rocky Mountains', 'MT': 'Rocky Mountains', 
    'UT': 'Rocky Mountains', 'WY': 'Rocky Mountains',
    'AK': 'Far West', 'CA': 'Far West', 'HI': 'Far West', 
    'NV': 'Far West', 'OR': 'Far West', 'WA': 'Far West',
    'AS': 'Outlying Areas', 'FM': 'Outlying Areas', 'GU': 'Outlying Areas', 
    'MH': 'Outlying Areas', 'MP': 'Outlying Areas', 'PR': 'Outlying Areas', 
    'PW': 'Outlying Areas', 'VI': 'Outlying Areas'
}

REGION_LIST = ['New England', 'Mid East', 'Great Lakes', 'Plains', 
               'Southeast', 'Southwest', 'Rocky Mountains', 'Far West', 'Outlying Areas']

# =============================================================================
# MARKET FIT SCORE WEIGHTS (Default - Grad Online Focus)
# =============================================================================

DEFAULT_SCORE_WEIGHTS = {
    'grad_focus': 0.30,        # Weight for graduate offerings/enrollment
    'online_intensity': 0.25,  # Weight for distance education
    'mobility_index': 0.20,    # Weight for out-of-state/foreign student draw
    'enrollment_scale': 0.15,  # Weight for total enrollment size
    'cost_accessibility': 0.10 # Weight for cost factors
}

# Alternative weight presets
WEIGHT_PRESETS = {
    'grad_online_default': {
        'name': 'Graduate Online Focus',
        'description': 'Prioritizes graduate programs and online enrollment',
        'weights': {
            'grad_focus': 0.30,
            'online_intensity': 0.25,
            'mobility_index': 0.20,
            'enrollment_scale': 0.15,
            'cost_accessibility': 0.10
        }
    },
    'ug_regional': {
        'name': 'Undergraduate Regional',
        'description': 'Emphasizes undergraduate enrollment and local market presence',
        'weights': {
            'grad_focus': 0.05,
            'online_intensity': 0.15,
            'mobility_index': 0.25,
            'enrollment_scale': 0.35,
            'cost_accessibility': 0.20
        }
    },
    'online_only': {
        'name': 'Online Market Focus',
        'description': 'Maximizes weight on distance education metrics',
        'weights': {
            'grad_focus': 0.15,
            'online_intensity': 0.50,
            'mobility_index': 0.20,
            'enrollment_scale': 0.10,
            'cost_accessibility': 0.05
        }
    },
    'national_reach': {
        'name': 'National Reach',
        'description': 'Prioritizes institutions with broad geographic draw',
        'weights': {
            'grad_focus': 0.15,
            'online_intensity': 0.20,
            'mobility_index': 0.40,
            'enrollment_scale': 0.15,
            'cost_accessibility': 0.10
        }
    }
}

# =============================================================================
# FILTER PRESETS
# =============================================================================

FILTER_PRESETS = {
    'grad_online_default': {
        'name': 'Grad Online Default',
        'description': 'Online/hybrid graduate program targeting',
        'filters': {
            'degree_lens': 'Graduate',
            'distance_ed': ['Exclusive DE', 'Some DE'],
            'award_levels': ['Masters', 'Doctoral'],
            'institution_level': ['4-year or above'],
            'min_enrollment': 500
        }
    },
    'ug_regional': {
        'name': 'UG Regional Default',
        'description': 'Traditional undergraduate regional targeting',
        'filters': {
            'degree_lens': 'Undergraduate',
            'distance_ed': ['Some DE', 'No DE'],
            'award_levels': ['Associates', 'Bachelors'],
            'institution_level': ['4-year or above', '2-year (less than 4)'],
            'min_enrollment': 1000
        }
    },
    'certificates_only': {
        'name': 'Certificate Programs',
        'description': 'Short-term credential programs',
        'filters': {
            'degree_lens': 'Both',
            'distance_ed': ['Exclusive DE', 'Some DE', 'No DE'],
            'award_levels': ['Certificates'],
            'institution_level': ['4-year or above', '2-year (less than 4)', 'Less than 2-year'],
            'min_enrollment': 100
        }
    },
    'online_scale': {
        'name': 'Large Online Programs',
        'description': 'High-volume online institutions',
        'filters': {
            'degree_lens': 'Both',
            'distance_ed': ['Exclusive DE'],
            'award_levels': ['Bachelors', 'Masters'],
            'institution_level': ['4-year or above'],
            'min_enrollment': 5000
        }
    }
}

# =============================================================================
# MARKET TAGS - Pre-defined market segments
# =============================================================================

MARKET_TAGS = {
    'high_online': {
        'name': 'High Online',
        'description': 'Institutions with >50% online enrollment',
        'criteria': {'pct_de_exclusive_min': 50}
    },
    'grad_heavy': {
        'name': 'Grad-Heavy',
        'description': 'Institutions where graduate enrollment > 40% of total',
        'criteria': {'grad_pct_min': 40}
    },
    'high_out_of_state': {
        'name': 'High Out-of-State',
        'description': 'Strong out-of-state student draw (>40%)',
        'criteria': {'out_of_state_pct_min': 40}
    },
    'affordable_public': {
        'name': 'Affordable Public',
        'description': 'Public institutions with below-median tuition',
        'criteria': {'control': [1], 'tuition_percentile_max': 50}
    },
    'research_doctoral': {
        'name': 'Research/Doctoral',
        'description': 'Institutions offering doctoral programs',
        'criteria': {'has_doctoral': True}
    },
    'hbcu': {
        'name': 'HBCU',
        'description': 'Historically Black Colleges and Universities',
        'criteria': {'hbcu': 1}
    },
    'community_colleges': {
        'name': 'Community Colleges',
        'description': 'Public 2-year institutions',
        'criteria': {'sector': [4]}
    }
}

# =============================================================================
# DISTANCE EDUCATION LEVEL CODES (ef2024a_dist.EFDELEV)
# =============================================================================

DE_LEVEL_CODES = {
    1: 'All students total',
    2: 'Undergraduate total',
    3: 'Undergraduate degree/certificate-seeking',
    4: 'First-time degree/certificate-seeking',
    5: 'Transfer-in degree/certificate-seeking',
    6: 'Continuing degree/certificate-seeking',
    7: 'Non-degree/certificate-seeking',
    11: 'Undergraduate other',
    12: 'Graduate total',
    19: 'Graduate degree/certificate-seeking',
    20: 'Graduate other',
}

# =============================================================================
# EFFY LEVEL CODES (effy2024.EFFYLEV/EFFYALEV)
# =============================================================================

EFFY_LEVEL_CODES = {
    1: 'All students',
    2: 'Undergraduate',
    3: 'Undergrad degree/cert seeking',
    4: 'First-time',
    5: 'Transfer-in',
    11: 'Continuing',
    12: 'Graduate',
    19: 'Non-degree/cert seeking',
    20: 'Unknown level',
}
