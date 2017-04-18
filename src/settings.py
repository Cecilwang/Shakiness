# Author: Cecil Wang (cecilwang@126.com)

import json
import utilities

class Settings(metaclass=utilities.singleton.SingletonMetaClass):

    def add_root(self, k, root, path):
        suffix = k.split('_')[-1]
        if suffix=='dir' or suffix=='path':
            path = root + path
        return path

    def __init__(self, settings_path='settings.json'):
        utilities.file.check_file_extension(settings_path, 'json')

        with open(settings_path) as settings_file:
            settings = json.load(settings_file)

        root = settings['root'][settings['platform']]

        for k, v in settings.items():
            if isinstance(v, dict):
                for in_k, in_v in v.items():
                    v[in_k] = self.add_root(in_k, root, in_v)
            else:
                v = self.add_root(k, root, v)
            setattr(self, k, v)


if __name__ == '__main__':
    print(Settings())
