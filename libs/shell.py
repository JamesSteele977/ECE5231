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

        self.to_kw = lambda arg: {
            'sen': 'sns', 
            'opt': 'opt', 
            'sol': 'sol'
        }.get(arg[:3].lower())
        
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

        for kw in ('sns', 'opt', 'sol'):
            saved = self._read_(kw)
            for name, config in saved.items():
                self._ini_(kw, name, config, verbose=False)
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

    def _ini_(self, kw: str, name: str, config: dict, verbose: bool = True) -> None:
        obj = self.subobjs[kw]['ini'](**config)
        desc = obj.desc
        desc['Name'] = name
        self.subobjs[kw]['saved'][name] = {
            'desc': desc,
            'obj': obj
        }
        if verbose:
            self._print_deschead(kw)
            self._print_descline(kw, desc)
        pass

    def _confirm(self, kw: str, name: str, head: str, msg: str) -> bool:
        confirm = str(input(f"{head} {self.subobjs[kw]['ini'].__name__} {name} {msg} [y/n] ")).lower()
        while confirm not in {'y', 'n'}:
            confirm = str(input(f"Invalid response '{confirm}' [y/n] ")).lower()
        if confirm == 'n':
            print('Exiting process...')
            return False
        return True

    def _delete_(self, kw: str, name: str) -> None:
        """
        Delete sensor/optimizer/solution object
        """
        name = name.lower()
        saved = self._read_(kw)

        in_json = name in saved
        in_subobjs = name in self.subobjs[kw]['saved']
        if not(in_json or in_subobjs):
            print(f"{self.subobjs[kw]['ini'].__name__} {name} does not exist.")
            return
        if not self._confirm(kw, name, 'Confirm action: Delete', '?'):
            return
        if in_json:
            del saved[name]
        if in_subobjs:
            del self.subobjs[kw]['saved'][name]
        self._write_(kw, saved)
        pass

    def _configure_(self, kw: str, name: str) -> None:
        """
        Configure old or create new sensor/optimzer object
        """
        if kw not in ('opt', 'sns'):
            print(f"Cannot configure object type: {self.subobjs[kw]['ini'].__name__}")

        name = name.lower()
        saved = self._read_(kw)
        if name in saved:
            if not self._confirm(kw, name, 'Warning:', 'already exists. Overwrite?'):
                return
            
        temp = self.settings['temp_config'][kw]
            
        while True:
            self._write_('tmp', temp)
            os.system(f"nano {self.settings['paths']['tmp']}")
            config = self._read_('tmp')
            if config != -1:
                break
    
        saved[name] = config
        self._write_(kw, saved)
        self._ini_(kw, name, config)
        pass

    def _save_solution(self, solution):
        pass

    """ GRAPHICAL """
    def _print_descline(self, kw: str, desc: dict) -> None:
        print('|'.join([self._pt(desc[key])[:l*8].ljust(l*8) for key, l in self.desc_cats[kw].items()]))
        pass

    def _print_deschead(self, kw: str) -> None:
        if len(self.subobjs[kw]['saved']) != 0:
            print(
                '|'.join([cat.ljust(l*8) for cat,l in self.desc_cats[kw].items()])\
                +'\n'+'-'*((sum(self.desc_cats[kw].values())+1)*8)
            )
        else:
            print("< none >")
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
    def do_list(self, arg):
        """
        Summary of all objects of given type registered in < Sensor Design Optimization Shell >
        <which>: Object type to list
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('which', help='Object type to list [Sensor, Optimizer, Solution]')
        try:
            args = parser.parse_args(arg.split())
            kw = self.to_kw(args.which)
            self._print_deschead(kw)
            for entry in self.subobjs[kw]['saved'].values():
                self._print_descline(kw, entry['desc'])
        except KeyError as e:
            print(e)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    def do_delete(self, arg):
        """
        Delete object saved in memory (Sensor, Optimizer, or Solution)
        <which>: Object type to delete
        <name>: Name of object to delete
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('which', help='Object type to list [Sensor, Optimizer, Solution]')
        parser.add_argument('name', help='Name of object to delete')
        try:
            args = parser.parse_args(arg.split())
            self._delete_(self.to_kw(args.which), args.name)
        except KeyError as e:
            print(e)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass

    def do_configure(self, arg):
        """
        Configure sensor object attributes
        <which>: Object type
        ["-n", "--name"] <name>: Name of object to configure
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('which', help='Object type to list [Sensor, Optimizer, Solution]')
        parser.add_argument('-n', '--name', help='Name of object to delete')
        try:
            args = parser.parse_args(arg.split())
            kw = self.to_kw(args.which)
            if args.name is None:
                args.name = f"{self.subobjs[kw]['ini'].__name__}_{len(self.subobjs[kw]['saved'])+1}"
            self._configure_(kw, args.name)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    def do_fit(self, arg):
        """
        Optimizes design specified in sensor object 
        <sensor>: Target sensor object
        ["-o", "--optimizer"] <optimizer>: Custom optimizer
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
    