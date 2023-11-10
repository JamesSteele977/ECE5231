import cmd, os, argparse, json
from typing import IO
from libs.sensor import Sensor
from libs.optim import Optim
import libs.proc as proc
import numpy as np
import tensorflow as tf

class ShellFn():
    def __init__(
            self, 
            cli_path = './cli',
            temp = 'temp.json',
            cfg = 'config.json',
            solve = 'solve.json',
            sens = 'sens.json',
            sensmd = 'sens.md',
            optim = 'optim.json',
            optimmd = 'optim.md'
        ) -> None:
        super().__init__()
        self.sensors = {}
        self.cli_path = cli_path
        jsons = ['temp', 'cfg', 'solve', 'sens', 'sensmd', 'optim', 'optimmd']
        cli = lambda x: os.path.join(cli_path, x)
        for var in jsons:
            setattr(self, var, cli(locals().get(var)))
        pass

    def _configure_sensor(self, name):
        """
        Configure sensor object's parameters and IO relationships
        """
        sensmd = proc.read_md(self.sensmd)
        temp_config = {
            'params': {
                '': []
            }, 
            'relations': [],
            'expressions': {},
            'footprint': "",
            'IO': "", 
            'README': sensmd
        } # Kwargs for Sensor() init

        while True:
            proc.write_json_dict(self.temp, temp_config)
            os.system(f"nano {self.temp}")
            config = proc.read_json(self.temp)
            if config != -1:
                break

        sens = proc.read_json(self.sens)

        if name in sens.keys():

            overwrite = str(input(f"Warning: Sensor {name} already exists. Overwrite? [y/n] ")).upper()
            while overwrite not in {'Y', 'N'}:
                overwrite = str(input(f"Invalid response {overwrite} [y/n] ")).upper()

            if overwrite.upper() == 'N':
                print('Discarding configuration file')
                return
            
        sens[name] = config
        proc.write_json_dict(self.sens, sens)
        
        new_sensor = Sensor(**config)
        self.sensors[name] = {
            'obj': new_sensor,
            'savepath': None,
            'IO': config['IO'],
            'footprint': config['footprint']
        }
        print(f"Sensor {name} configured")
        pass

    def _configure_optim(self, name: str):
        self.optims[name] = Optim()
        pass

    def _save_solution(self, solution):
        pass

    """ GEN """
    def _rectify_tab(self, x, n: int) -> str:
        tlen = (8*n+1)
        x = str(x)[:tlen]
        return x+(' '*(tlen-len(x)))

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
        parser = argparse.ArgumentParser(description='Initialize new sensor object')
        parser.add_argument("-n", "--name", default=f"Sensor_{len(self.sensors)+1}")
        try:
            args = parser.parse_args()
            self._configure_sensor(args.name)
            print(f"-----{args.name}-----\n{self.sensors[args.name]['obj'].summary()}")
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
            args = parser.parse_args()
            if args.sensor is None:
                print()
            else:
                self._configure_sensor(args.sensor)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    def do_sensors(self, arg):
        """
        Summary of all sensors registered in < Sensor Design Optimization Shell >
        """
        cats = {'Name':2, 'Saved':1, 'Parameters':4, 'IO':2, 'Footprint':2}
        print('|'.join([self._rectify_tab(cat,len) for cat,len in cats.items()]))
        print('-'*(sum(cats.values())*8))
        for name, info in self.sensors.items():
            train_vars = self._rectify_tab(
                ','.join([var.name for var in info['obj'].trainable_variables]),
                cats['Parameters']
            )
            print('|'.join([
                self._rectify_tab(name, cats['Name']),
                self._rectify_tab(info['savepath'], cats['Saved']),
                train_vars,
                self._rectify_tab(info['IO'], cats['IO']),
                self._rectify_tab(info['footprint'], cats['Footprint'])
            ]))
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
            solution = self.sensors[args.sensor].fit()
            if args.save:
                self._save_solution(solution)
        except KeyError:
            print(f"Error: Sensor {args.sensor} does not exist")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass
    
    # GENERAL
    def do_view(self, arg) -> None:
        pass

    