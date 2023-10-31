import cmd, os, argparse, json
from typing import IO
from libs.sensor import Sensor
from libs.optim import Optim
import numpy as np

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
        with open(os.path.join(self.sensmd), 'r') as f:
            sensmd = f.read()

        temp_config = {
            'parameters': [], 
            'expressions': {},
            'constraints': {},
            'footprint': "",
            'IO': "", 
            'README': sensmd
        } # Kwargs for Sensor() init

        while True:
            temp_path = os.path.join(self.cli_path, 'temp.json') # Path to file for nano edit
            with open(temp_path, 'w') as f:
                json.dump(temp_config, f, indent=4) # Overwrite temp file with temp_config

            os.system(f"nano {temp_path}") # User input for Sensor() configuration
            with open(temp_path, 'r') as f:
                try:
                    config = json.load(f)
                    break
                except json.JSONDecodeError as e:
                    print(e)

        with open(os.path.join(self.cli_path, 'sens.json'), 'w') as f:
            sens = json.load(f) # Get current sensor registry

            if name in sens.keys():
                overwrite = str(input(f"Warning: Sensor {name} already exists. Overwrite? [y/n] ")).upper()
                while overwrite not in {'Y', 'N'}:
                    overwrite = str(input('Invalid response [y/n]')).upper()
                if overwrite.upper() == 'N':
                    print('Discarding configuration file')
                    return None
            
            sens[name] = config
            json.dump(sens, f)
        
        new_sensor = Sensor(**config)
        self.sensors[name] = new_sensor
        print(f"Sensor {name} configured successfully! Optim-Ready: {new_sensor.ready['bool']}")
        if not new_sensor.ready['bool']:
            print(f"Sensor {name} missing args: {new_sensor.ready['list']}")

        pass

    def _configure_optim(self, name):
        self.optims[name] = Optim()
        pass

    def _save_solution(self, solution):
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
    def do_clear(self, arg) -> None:
        """
        Clear the terminal
        """
        if os.name == 'nt':  # For Windows
            os.system('cls')
        else:  # For macOS and Linux
            os.system('clear')
        pass

    def do_exit(self, arg) -> None:
        """
        Exit the shell
        """
        return True

    def do_readme(self, arg) -> None:
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
    def do_init_sensor(self, arg) -> None:
        """
        Initializes new sensor object
        ["-n", "--name"] <name>: Optional name for sensor
        """
        parser = argparse.ArgumentParser(description='Initialize new sensor object')
        parser.add_argument("-n", "--name", default=f"Sensor_{len(self.sensors)+1}")
        try:
            args = parser.parse_args()
            self._configure_sensor(args.name)
            print(f"-----{args.name} sucessfully created-----\n{self.sensors[args.name].summary()}")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_configure(self, arg) -> None:
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
    
    def do_sensors(self, arg) -> None:
        """
        Summary of all sensors registered in < Sensor Design Optimization Shell >
        """
        tab = lambda x: x*'\t'
        print(f"Name{tab(2)}|Saved{tab(2)}|Parameters{tab(2)}|IO{tab(2)}\n{'-'*32}")
        for name, info in self.sensors.items():
            print(f"{name[:17]}\t|{info['savepath']}")
        pass
    
    # OPTIMIZER
    def do_optim(self, arg) -> None:
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

    def do_sensors(self, arg) -> None:
        """
        Summary of all sensors registered in < Sensor Design Optimization Shell >
        """
        tab = lambda x: x*'\t'
        print(f"Name{tab(2)}|Saved{tab(2)}|Parameters{tab(2)}|IO{tab(2)}\n{'-'*32}")
        for name, info in self.sensors.items():
            print(f"{name[:17]}\t|{info['savepath']}")
        pass
    
    # GENERAL
    def do_view(self, arg) -> None:
        pass

    