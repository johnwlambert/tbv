# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""Datetime utility functions.

Ref: https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/datetime_utils.py
"""

import datetime


def generate_datetime_string() -> str:
    """Generate a formatted datetime string.

    Returns:
        String with of the format YYYY_MM_DD_HH_MM_SS with 24-hour time used
    """
    return f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"