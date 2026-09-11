"""
idempotent-core: Unified Native C++20 and CUDA Zero-Copy Idempotent Engine.

Protected under U.S. Patent Application No. 64/148,668 ("Patent Pending").
Author: Dr. A. Emre ÇETİN <aemre.cetin@gmail.com>
"""

from .binding import (
    compact_inplace,
    generate_idempotent_map,
    get_native_version
)

__version__ = "1.1.0"
__author__ = "Dr. A. Emre ÇETİN"
__email__ = "aemre.cetin@gmail.com"
__license__ = "Apache-2.0 WITH Patent-Evaluation-Grant"

__all__ = [
    "compact_inplace",
    "generate_idempotent_map",
    "get_native_version",
]