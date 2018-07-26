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


"""
Regex pattern for ECG beat annotations used for extracting single beats.
Should be compiled with re.VERBOSE.
"""
BEAT_ANNOTATIONS_PATTERN = r'''
    (?P<p_start>
        \(
    )
    (?P<p>
        p
    )
    (?P<p_end>
        \)
    )
    (?P<r_start>
        \(
    )
    (?P<r>
        [NVSFQLRBAaJrFejnE/f]
    )
    (?P<r_end>
        \)
    )
    (?P<t_start>
        \(
    )
    (?P<t>
        t+
    )
    (?P<t_end>
        \)
    )
'''
