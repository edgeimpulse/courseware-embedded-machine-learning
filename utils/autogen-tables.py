#!/usr/bin/env python3

"""
Autogenerate tables from filenames

Script to generate markdown tables from filenames in a given folder. This script is specific to this project in that the filenames are parsed with decimals as delimiters.

[module].[section].[resource].resource-name.[attribution].[extention]

Example call:
python autogen-tables.py -d [path/to/directory]

License: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""

import argparse
import os

# Authorship
__author__ = "Shawn Hymel"
__copyright__ = "Copyright 2022, Edge Impulse"
__license__ = "Apache 2.0"
__version__ = "1.0"

################################################################################
# Settings

repo_url = "https://github.com/edgeimpulse/course-embedded-machine-learning/raw/main/"
attr_urls = [
    "https://github.com/edgeimpulse/courseware-embedded-machine-learning#1-slides-and-written-material-for-introduction-to-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40",
    "https://github.com/edgeimpulse/courseware-embedded-machine-learning#2-slides-and-written-material-for-computer-vision-with-embedded-machine-learning-by-edge-impulse-is-licensed-under-cc-by-nc-sa-40",
    "https://github.com/edgeimpulse/courseware-embedded-machine-learning#3-slides-and-written-material-for-tinyml-courseware-by-tinymlx-is-licensed-under-cc-by-nc-sa-40"
]



################################################################################
# Main

# Script arguments
parser = argparse.ArgumentParser(description="Script that looks through "
            "filenames in given directory and auto-generates markdown "
            "tables and links")
parser.add_argument('-d',
                   '--directory',
                   action='store',
                   dest='dir_path',
                   type=str,
                   required=True,
                   help="Directory containing the files you want to list in "
                        "the table.")

# Parse the arguments
args = parser.parse_args()
dir_path = args.dir_path

# Get files in directory (ignore subdirectories)
filenames = []
for f in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, f)):
        filenames.append(f)

# Go through files
for f in filenames:

    # Get ID and description
    parsed = f.split('.')
    id = str(parsed[0]) + "." + str(parsed[1]) + "." + str(parsed[2])
    desc = parsed[3].replace('-', ' ').capitalize()
    
    # Construct link based on file type
    link = repo_url + dir_path + '/' + f
    link = link.replace(' ', '%20')
    if parsed[5] == 'docx':
        link = "[doc](" + link + ")"
    elif parsed[5] == 'pptx':
        link = "[slides](" + link + ")"
    else:
        print("ERROR: Unsupported file type: " + f)
        exit()
    
    # Create attribution link
    attr = parsed[4]
    if not attr.isnumeric():
        print("ERROR: Attribution must be a number: " + f)
        exit()
    attr = int(attr)
    if (attr > 0) and (attr <= len(attr_urls)):
        attr = "[[" + str(attr) + "]](" + attr_urls[attr - 1] + ")"
    else:
        attr = ""
    
    # Print table entry
    print("| " + id + " | " + desc + " | " + link + " | " + attr + " |")