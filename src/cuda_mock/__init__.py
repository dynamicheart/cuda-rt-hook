from .cuda_mock_impl import add
from .cuda_mock_impl import initialize
from .cuda_mock_impl import internal_install_hook
from .cuda_mock_impl import xpu_initialize
import os

def exec_shell(cmd, print_result = False):
    with os.popen(cmd) as pf:
        lines = pf.readlines()
    if print_result:
        for line in lines:
            print(line.strip())
    return ''.join(lines)


class DynamicObj:
    def __init__(self, file):
        with open(file, "rt") as f:
            lines = f.readlines()
            self.code = ''.join(lines)
        self.compile_opts = []
    def compile(self):
        if not self.check_env():
            raise Exception("can't find host compiler!")
        from datetime import datetime
        current_time = datetime.now()
        current_time = str(current_time).replace(':', '_')
        current_time = str(current_time).replace(' ', '_')
        current_time = str(current_time).replace('.', '_')
        
        file_name = f'__cuda_rt_hook__{current_time}'
        cpp_file_name = f'/tmp/{file_name}.cpp'

        with open(cpp_file_name, 'wt') as f:
            f.write(self.code)

        self.lib_file_name = f'/tmp/{file_name}.so'

        opts = ' '.join(self.compile_opts)
        compile_cmd = f'{self.compiler} -shared -fpic {cpp_file_name} {opts} -o {self.lib_file_name}'
        exec_shell(compile_cmd, True)
        return self

    def check_env(self):
        if len(self.get_compiler('gcc')) == 0:
            self.compiler = "gcc"
            return True
        if len(self.get_compiler('clang')) == 0:
            self.compiler = "clang"
            return True
        return False
        
    def get_compiler(self, name):
        result = exec_shell(f'which {name}').strip()
        if 0 == len(result):
            return ''
        if -1 == result.find("not found"):
            return ''
        return result

    def appen_compile_opts(self, *args):
        self.compile_opts = list(args)
        return self
    
    def get_lib(self):
        return self.lib_file_name

initialize()
