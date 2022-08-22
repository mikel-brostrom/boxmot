#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The code is based on
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
# Copyright (c) OpenMMLab.

import os.path as osp
import shutil
import sys
import tempfile
from importlib import import_module
from addict import Dict


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config(object):

    @staticmethod
    def _file2dict(filename):
        filename = str(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                shutil.copyfile(filename,
                                osp.join(temp_config_dir, '_tempconfig.py'))
                sys.path.insert(0, temp_config_dir)
                mod = import_module('_tempconfig')
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules['_tempconfig']
        else:
            raise IOError('Only .py type are supported now!')
        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)
