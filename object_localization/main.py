"""
This script contains the code to prepare the data for training on the object localization task
"""

import os
import xml.etree.ElementTree as ET

from typing import Union, Tuple, List
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf


