#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from d2go.config import CfgNode as CN

def add_aut_config(cfg):
    """
    Add config for SemiSupSegRunner.
    """
    _C = cfg
    #New added for discriminator
    _C.UNBIASEDTEACHER.DIS_LOSS_WEIGHT = 0.1
    _C.UNBIASEDTEACHER.DIS_TYPE = "concate" #["concate","p2","multi"]
    _C.UNBIASEDTEACHER.ISAUG = "Yes"
