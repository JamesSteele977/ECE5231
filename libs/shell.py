import cmd, os, argparse
from typing import IO
from libs.sensor import Sensor
from libs.optim import Optim

class ShellFn():
    def __init__(self, cli_path) -> None:
        super().__init__()
        self.sensors = {}
        self.cli_path = cli_path
        pass

    def _configure_sensor(self, name):
        self.sensors[name] = Sensor()
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
        cli_path: str = './cli'
    ) -> None:
        cmd.Cmd.__init__(self, completekey, stdin, stdout)
        ShellFn.__init__(self, cli_path)
        pass

    intro: str = '< Sensor Design Optimization Shell >'
    prompt: str = 'optimtool$ '

    def completedefault(self, text, line, start_idx, end_idx):
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
            print(f"-----{args.name} sucessfully created-----\n{self.sensors[args.name].summary()}")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_configure(self, arg):
        """
        Configure sensor object attributes
        ["-s", "--sensor"] <name>: Target sensor object
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--sensor', choices=list(self.sensors.keys()))
        try:
            args = parser.parse_args()
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
        tab = lambda x: x*'\t'
        print(f"Name{tab(2)}|Saved{tab(2)}|Parameters{tab(2)}|IO{tab(2)}\n{'-'*32}")
        for name, info in self.sensors.items():
            print(f"{name[:17]}\t|{info['savepath']}")
        pass
    
    # OPTIMIZER
    def do_optim(self, arg):
        """
        Optimizes design specified in sensor object 
        ["-s", "--sensor"] <name>: Target sensor object
        ["-o", "--optimizer"] <name>: Custom optimizer
        ["-sv", "--save"]: Save optimization results, accessable through "open" with ["-sol, --solution"] flag
        """ 
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--sensor')
        parser.add_argument('-o', '--optimizer')
        parser.add_argument('-sv', '--save', action='store_true')
        try:
            args = parser.parse_args()
            solution = self.sensors[args.sensor].fit()
            if args.save:
                self._save_solution(solution)
        except KeyError:
            if args.sensor is None:
                print('Error: No sensor argument provided')
            else:
                print(f'Error: Sensor {args.sensor} does not exist')
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_sensors(self, arg):
        """
        Summary of all sensors registered in < Sensor Design Optimization Shell >
        """
        tab = lambda x: x*'\t'
        print(f"Name{tab(2)}|Saved{tab(2)}|Parameters{tab(2)}|IO{tab(2)}\n{'-'*32}")
        for name, info in self.sensors.items():
            print(f"{name[:17]}\t|{info['savepath']}")
        pass
    
    # GENERAL
    def do_view(self, arg):
        pass

    