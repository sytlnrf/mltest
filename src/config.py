# coding=utf-8
import sys
from os.path import dirname

def get_project_dir():
    """
    get the project dir
    """
    return dirname(dirname(__file__))
