from __future__ import print_function
import hickle as hkl
import json
import os
from util import log
import re
from IPython import embed


class ModelSaver:
    @classmethod
    def from_dict(cls, dict):
        print (cls.PARAMS)
        param_dict = {k: dict[k] for k in cls.PARAMS}
        print(param_dict)
        return cls(**param_dict)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.PARAMS}

    @classmethod
    def load_from_file(cls, path):
        with open(path, 'r') as f:
            data = hkl.load(f)
        return cls.from_dict(data)

    def save_result(self, result_json, path):
        key_list = sorted(result_json.keys())
        with open(path, 'w') as f:
            for key in key_list:
                result = result_json[key]
                print(result, file=f)
        log.info("Save to {}".format(path))

    def save_to_file(self, attr, path):
        param = attr
        with open(path, 'w') as f:
            hkl.dump(self.to_dict(), f)
        path = os.path.splitext(path)[0] + '.json'
        with open(path, 'w') as f:
            json.dump(param, f)
        log.info("Save parame to {}".format(path))

    def print_params(self):
        print("")
        for key, value in self.to_dict().iteritems():
            if key != 'word_embed':
                print("{} : {}".format(key, value))
        print("")
