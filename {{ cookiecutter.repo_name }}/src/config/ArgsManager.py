"""
プログラムの引数渡しで変更する可能性のある定数(int,str,float,bool)はargs.hogehoge(default_value, --arg_name)でラップすることで、
通常の実行(python hogehoge.py)ではdefault_valueを返し、引数を取れば(python hogehoge.py --arg_name overwrite_value)上書きする。
今後は、プログラム実行はスクリプトから行うようにするが、
例えば、CUDA指定や、EPOCH数や、データパス程度なら渡したい。
(データセットが異なる場合、ドメインをきるのはいいけど、関数の再作成とかまじで無駄なのでどうにかしたい。)

# 基本ルール

const = args.hoge(default_const, arg_name)

arg_nameは--arg_name。短縮形は作らない。--small_sname_case。

int_const = args.int(32, '--arg_name')
str_const = args.str("cuda", '--arg_name')
など。

config.const.pyで宣言したargs.hogehoge()でも、
config.domain.const.pyで再宣言可能。
この場合、再宣言したdefaultがconfig.constより優先されるが、
arg指定があれば、arg指定が最優先される。
ただし、一部のpost_setting枠に関しては、必要に応じて、argsが動的に変化する場合がある。
(post_setting.py参考)

#
python hoge.py --arg_name value
でarg_name上書きされる。

argsの強さ
config.constでの宣言 < config.[domain].constでの(再)宣言 < arg引数

config.constやconfig.[domain].constでの宣言例:
```config.const.py file
DEVICE = args.str("cuda", "--device")
```

args naming rule
should be declarative. using small_snake_case. never use abbr. if you use any abbr, write this on below.

# Abbriviation rule used in this project

abbr = abbreviation
"""
import argparse
from distutils.util import strtobool


class ArgSetting:
    def __init__(self, default, key, type_class):
        self.default = default
        self.key = key
        self.type = type_class

    def __eq__(self, other):
        return self.key == other.key and self.type == other.type


class ArgsManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = list()

    def get_arg(self, arg_setting):
        args = self.parser.parse_args()
        args_value = args.__dict__[arg_setting.key.replace("-", "")]
        if args_value != arg_setting.default:
            if arg_setting.type != bool:
                return args_value
            else:
                return strtobool(args_value)
        else:
            return arg_setting.default

    def overwritable(self, default, key, type_class):
        setting = ArgSetting(default, key, type_class)
        if setting not in self.args:
            self.args.append(setting)
            if type_class != bool:
                self.parser.add_argument(key, type=type_class, default=default)
            else:
                self.parser.add_argument(key, type=str, default="True" if default else "False")
            return self.get_arg(setting)
        else:
            old_setting = self.args[self.args.index(setting)]
            old_default = old_setting.default
            self.args[self.args.index(setting)] = setting

            if self.get_arg(old_setting) == old_default:
                return setting.default
            else:
                return self.get_arg(setting)

    def str(self, default, key):
        return self.overwritable(default, key, str)

    def int(self, default, key):
        return self.overwritable(default, key, int)

    def float(self, default, key):
        return self.overwritable(default, key, float)

    def bool(self, default, key):
        return self.overwritable(default, key, bool)


args = ArgsManager()
