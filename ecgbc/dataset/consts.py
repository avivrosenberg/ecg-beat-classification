"""
This module contains constants used by the dataset package.
"""

"""
Extension of WFDB header files.
"""
WFDB_HEADER_EXT = '.hea'

"""
Regex pattern that heuristically detects ECG channels in WFDB records.
"""
ECG_CHANNEL_PATTERN = r'ECG|lead\si+|MLI+|v\d|\bI+\b'
