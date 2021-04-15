#!/usr/bin/python

import sys, re, os

def edit():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, str(sys.argv[1]))
    yaml = "---\nlayout: post\ntitle: TITLE\nmathjax: true\ncategories:\n  - category\ntags:\n  - tag\n---\n\n"
    with open(path, 'r') as file:
        filedata = file.read()
    filedata = re.sub(r"!\[png\]\(", "<img src=\"/images/", filedata)
    filedata = re.sub(".png\)", ".png\">", filedata)
    filedata = yaml + filedata
    with open(path, 'w') as file:
        file.write(filedata)

if __name__ == '__main__':
    edit()