import cmd, os, argparse, json
from typing import IO
from libs.sensor import Sensor
from libs.optim import Optim, Solve
import libs.proc as proc
import numpy as np
import tensorflow as tf

class ShellFn():
    def __init__(self, settings: str = './settings.json') -> None:
        self.settings = proc.read_json(settings)[ShellFn.__name__]
        self.subobjs = {
            kw: {'ini': _class, 'saved': {}} for kw, _class in zip(
                ('sns', 'opt', 'sol'), (Sensor, Optim, Solve)
            )
        }
        
        for kw in ('sns', 'opt', 'sol'):
            data = self._read_(kw)
            for name, config in data.items():
                self._ini_(kw, name, config)

        self.desc_cats = {
            'sns': {'Name':2, 'Parameters':4, 'IO':2, 'Footprint':2},
            'opt': {},
            'sol': {}
        }

        self._pt = lambda x: self._pt_cases[str(type(proc.deref_iter(x)))](proc.deref_iter(x))
        self._clstr = lambda x: f"<class '{x}'>"
        self._pt_cases = {
            self._clstr('str'): lambda x: x,
            self._clstr('float'): lambda x: str(x),
            self._clstr('int'): lambda x: str(x),
            self._clstr('dict'): lambda x: ' '.join([f"{self._pt(y)}:{self._pt(z)}" for y, z in x.items()]),
            self._clstr('tuple'): lambda x: ','.join([self._pt(y) for y in x])
        }
        pass

    def _check_(self, kw: str) -> None:
        if kw not in self.settings['paths'].keys():
            raise KeyError(f"Invalid get keyword")
        path = self.settings['paths'][kw]
        if not os.path.exists(path):
            split = os.path.join(*os.path.split(path)[:-1])
            if not os.path.exists(split):
                os.mkdir(split)
            with open(path, 'w') as f:
                json.dump({}, f)
        elif not os.path.isfile(path):
            os.rmdir(path)
            with open(path, 'w') as f:
                json.dump({}, f)
        return path

    def _read_(self, kw: str) -> dict:
        return proc.read_json(self._check_(kw))

    def _write_(self, kw: str, data: dict) -> None:
        proc.write_json(self._check_(kw), data)
        pass

    def _ini_(self, kw: str, name: str, config: dict) -> None:
        obj = self.subobjs[kw]['ini'](**config)
        desc = obj.desc
        desc['Name'] = name
        self.subobjs[kw]['saved'][name] = {
            'desc': desc,
            'obj': obj
        }
        pass

    def _configure_sensor(self, name: str) -> None:
        """
        Configure sensor object's parameters and IO relationships
        """
        sns = self._read_('sns')
        if name in sns:
            overwrite = str(input(f"Warning: Sensor {name} already exists. Overwrite? [y/n] ")).upper()
            while overwrite not in {'Y', 'N'}:
                overwrite = str(input(f"Invalid response '{overwrite}' [y/n] ")).upper()
            if overwrite.upper() == 'N':
                print('Exiting process...')
                return
            
        while True:
            self._write_('tmp', self.settings['temp_config'])
            os.system(f"nano {self.settings['paths']['tmp']}")
            config = self._read_('tmp')
            if config != -1:
                break
    
        sns[name] = config
        self._write_('sns', sns)
        self._ini_('sns', name, config)
        pass

    def _configure_optim(self, name: str):
        self.optims[name] = Optim()
        pass

    def _save_solution(self, solution):
        pass

    """ GRAPHICAL """
    def _print_descline(self, kw: str, desc: dict) -> None:
        print('|'.join([self._pt(desc[key])[:l*8].ljust(l*8) for key, l in self.desc_cats[kw].items()]))
        pass

class UI(cmd.Cmd, ShellFn):
    def __init__(
        self,
        completekey: str = 'tab', 
        stdin: IO[str] | None = None, 
        stdout: IO[str] | None = None,
    ) -> None:
        cmd.Cmd.__init__(self, completekey, stdin, stdout)
        ShellFn.__init__(self)
        pass

    intro: str = '< Sensor Design Optimization Shell >'
    prompt: str = 'optimtool$ '

    def completedefault(self, text, line, start_idx, end_idx) -> list:
        # This method is called for tab completions not handled by other methods
        if not text:
            return self.commands
        else:
            return [command for command in self.commands if command.startswith(text)]

    # -------------------------------------------------------------------------------
    """ UI/NAV COMMANDS """
    def do_clear(self, arg):
        """
        Clear the terminal
        """
        if os.name == 'nt':  # For Windows
            os.system('cls')
        else:  # For macOS and Linux
            os.system('clear')
        pass

    def do_exit(self, arg):
        """
        Exit the shell
        """
        return True

    def do_readme(self, arg):
        """
        Detailed description of how to use < Sensor Design Optimization Shell >
        """
        if 'README.md' in os.listdir('./'):
            with open('./README.md', 'r') as file:
                print(file.read())
        else:
            print("README.md not found")
        pass

    # -------------------------------------------------------------------------------
    """ OBJECT COMMANDS """
    # SENSOR
    def do_init_sensor(self, arg):
        """
        Initializes new sensor object
        ["-n", "--name"] <name>: Optional name for sensor
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--name", default=f"Sensor_{len(self.subobjs['sns']['saved'])+1}")
        try:
            args = parser.parse_args()
            self._configure_sensor(args.name)
            self._print_descline(args.name)
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_configure(self, arg):
        """
        Configure sensor object attributes
        <sensor>: Target sensor object
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('sensor')
        try:
            args = parser.parse_args(arg.split())
            self._configure_sensor(args.sensor)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    def do_list(self, arg):
        """
        Summary of all sensors registered in < Sensor Design Optimization Shell >
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('which')
        to_kw = lambda arg: {'sensors': 'sns', 'optimizers': 'opt', 'solutions': 'sol'}.get(arg)
        try:
            args = parser.parse_args(arg.split())
            kw = to_kw(args.which)
            if len(self.subobjs[kw]['saved']) != 0:
                print(
                    '|'.join([cat.ljust(l*8) for cat,l in self.desc_cats[kw].items()])\
                    +'\n'+'-'*((sum(self.desc_cats[kw].values())+1)*8)
                )
                for entry in self.subobjs['sns']['saved'].values():
                    self._print_descline(kw, entry['desc'])
            else:
                print("< none >")
        except KeyError as e:
            print(e)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    # OPTIMIZER
    def do_optim(self, arg):
        """
        Optimizes design specified in sensor object 
        <sensor>: Target sensor object
        ["-o", "--optimizer"] <optimizer>: Custom optimizer
        ["-sv", "--save"]: Save optimization results, accessable through "open" with ["-sol, --solution"] flag
        """ 
        parser = argparse.ArgumentParser()
        parser.add_argument('sensor')
        parser.add_argument('-o', '--optimizer')
        parser.add_argument('-sv', '--save', action='store_true')
        try:
            args = parser.parse_args()
            solution = self.subobjs['sns'] [args.sensor].fit()
            if args.save:
                self._save_solution(solution)
        except KeyError:
            print(f"Error: Sensor {args.sensor} does not exist")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass
    