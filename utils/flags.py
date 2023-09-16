"""Flag registry."""

import argparse

PARSER = argparse.ArgumentParser(description='TS Flex')
CLASSES = []


def with_flags(cls):
    cls.register_flags(PARSER)
    CLASSES.append(cls)
    return cls


def global_parse():
    args = PARSER.parse_args()
    for cls in CLASSES:
        cls.args = args
