import matplotlib.pyplot as plt

from .sensor import Sensor, SensorConfig
from .optim import Optim, OptimConfig
from .solution import Solution, StateVariable, SolutionSave
import libs.proc as proc
import cmd, os, argparse, json

from enum import Enum
from dataclasses import dataclass, fields, asdict
from typing import IO, Dict, Tuple, List

class ShellPath(Enum):
    TEMP: str = './cli/temp.json'

    SENSORS: str = './cli/sensors.json'
    OPTIMIZERS: str = './cli/optimizers.json'
    SOLUTIONS: str = './cli/solutions.json'

class Subobject(Enum):
    SENSOR: int = ShellPath.SENSORS
    OPTIMIZER: int = ShellPath.OPTIMIZERS
    SOLUTION: int = ShellPath.SOLUTIONS

tabs_to_spaces: callable = lambda x: x * 8
DescriptionFormat: Dict[int, Dict[str, int]] = {
    Subobject.SENSOR: {'Name':tabs_to_spaces(2), 'Trainable Variables':tabs_to_spaces(4), 'IO':tabs_to_spaces(2), 'Bandwidth':tabs_to_spaces(2)}, 
    Subobject.OPTIMIZER: {'Name':tabs_to_spaces(2)},
    Subobject.SOLUTION: {'Name':tabs_to_spaces(2)}
}

class ShellFn():
    def __init__(self) -> None:
        self.saved_objects: Dict[Subobject, Dict[str, Sensor | Optim | Solution]] = {
            Subobject.SENSOR: {},
            Subobject.OPTIMIZER: {},
            Subobject.SOLUTION: {}
        }
        
        # Reinitialize saved objects from cli json files
        for subobject in Subobject.__members__:
            saved: dict = self._read(subobject)
            for name, config in saved.items():
                self._save_object(subobject, name, config, verbose=False)
        pass

    """ UI """
    def _string_to_kw(self, string: str) -> Subobject:
        match string[:3].upper():
            case 'SOL':
                return Subobject.SOLUTION
            case 'OPT':
                return Subobject.OPTIMIZER
            case 'SEN':
                return Subobject.SENSOR
        raise KeyError(f"Invalid object type {string}. Choose from: 'Solution', 'Optimizer', 'Sensor'")

    def _user_confirm(self, kw: Subobject, name: str, head: str, msg: str) -> bool:
        confirm: str = str(input(f"{head} {kw.name} {name} {msg} [y/n] ")).upper()
        while confirm not in {'Y', 'N'}:
            confirm: str = str(input(f"Invalid response '{confirm}' [y/n] ")).upper()
        if confirm == 'N':
            print('Exiting process...')
            return False
        return True

    """ READ/WRITE JSON RECORDS """
    def _read(self, kw: Subobject | ShellPath) -> dict:
        self._check_rectify_path(kw)
        return proc.read_json(kw.value)

    def _write(self, kw: Subobject | ShellPath, data: dict) -> None:
        self._check_rectify_path(kw)
        proc.write_json(kw.value, data)
        pass
    
    def _check_rectify_path(self, kw: Subobject | ShellPath) -> None:
        if not os.path.exists(kw.value):
            split = os.path.join(*os.path.split(kw.value)[:-1])
            if not os.path.exists(split):
                os.mkdir(split)
        elif not os.path.isfile(kw.value):
            os.rmdir(kw.value)
        with open(kw.value, 'w') as f:
            json.dump({}, f)
        pass    

    """ CONFIGURE REFACTORING """
    def _get_temp_json_dataclass_dict(self, dataclass: dataclass) -> dict:
        dataclass_dict: dict = {}
        for field in fields(dataclass):
            match field.type:
                case str() | int() | float():
                    val: str = ''
                case dict():
                    val: dict = {}
                case tuple() | list() | set():
                    val: list = []
                case _:
                    print('unknown type enountered in config') #--------------------------------------------------------------------------------------------
            dataclass_dict[field.name] = val
        return dataclass_dict

    def _save_object(self, kw: Subobject, name: str, config: dict, verbose: bool = True) -> None:
        self.saved_objects[kw][name]: dict = config
        if verbose:
            self._print_description_header(kw)
            self._print_description_line(config)
        pass

    """ LIST REFACTORING """
    def _recursively_format_obj_to_str(self, obj: str | tuple | set | list | dict | int | float) -> str:
        match obj:
            case str():
                return obj
            case tuple() | set() | list():
                return ','.join([self._recursively_format_obj_to_str(i) for i in obj])
            case dict():
                return ' '.join(f"{self._recursively_format_obj_to_str(key)}:{self._recursively_format_obj_to_str(label)}" for key, label in obj.items())
            case _:
                return str(obj)
            
    def _print_description_line(self, name: str, config: dict) -> None:
        print(f"{name.ljust(24)}|{'|'.join([self._recursively_format_obj_to_str(label)[:24].ljust(24) for label in config.values()])}")
        pass

    def _print_config_fields(self, config_type: type) -> None:
        print(f"{'Name'.ljust(24)}|{'|'.join([field.name.ljust(24) for field in fields(config_type)])}\n{'-'*((len(fields(config_type)) * 24) + 4)}")
        pass

    def _print_description_header(self, kw: Subobject) -> None:
        if len(self.saved_objects[kw]) != 0:
            match kw:
                case Subobject.SENSOR:
                    self._print_config_fields(SensorConfig)
                case Subobject.OPTIMIZER:
                    self._print_config_fields(OptimConfig)
                case Subobject.SOLUTION:
                    self._print_config_fields(SolutionSave)
                    
        else:
            print("< none >")
        pass

    """ MAIN USER FUNCTIONS """
    def _list_(self, kw: Subobject):
        self._print_description_header(kw)
        for name, entry in self.saved_objects[kw].items():
            self._print_description_line(kw, name, entry)
        pass

    def _delete_(self, kw: Subobject, name: str) -> None:
        """
        Delete sensor/optimizer/solution object
        """
        name: str = name.upper()
        saved: dict = self._read(kw)

        in_json: bool = name in saved
        in_saved_objects: bool = name in self.saved_objects[kw]

        if not(in_json or in_saved_objects):
            print(f"{name} does not exist.")
            return
        
        if not self._user_confirm(kw, name, 'Confirm action: Delete', '?'):
            return
        
        if in_json:
            del saved[name]
        if in_saved_objects:
            del self.saved_objects[kw][name]
        
        self._write(kw, saved)
        pass

    def _configure_(self, kw: Subobject, name: str) -> None:
        """
        Configure old or create new sensor/optimzer object
        """
        if kw == Subobject.SOLUTION:
            print(f"Cannot configure object type: Solution")
            return

        name: str = name.upper()
        saved: dict = self._read(kw)
        if name in saved:
            if not self._user_confirm(kw, name, 'Warning:', 'already exists. Overwrite?'):
                return

        match kw:
            case Subobject.SENSOR:
                temp: dict = self._get_temp_json_dataclass_dict(SensorConfig)
            case Subobject.OPTIMIZER:
                temp: dict = self._get_temp_json_dataclass_dict(OptimConfig)
            
        while True:
            self._write(ShellPath.TEMP, temp)
            os.system(f"nano {ShellPath.TEMP}")
            config: dict | int = self._read(ShellPath.TEMP)
            if config != -1:
                break
    
        saved[name]: dict = config
        self._write(kw, saved)
        self._save_object(kw, name, config)
        pass

    def _fit_(self, optimizer: str, sensor: str) -> None:

        sensor: Sensor = Sensor(self.saved_objects[Subobject.SENSOR][sensor])
        optimizer: Optim = Optim(
            self.saved_objects[Subobject.OPTIMIZER][optimizer],
            sensor.sensor_profile
        )
        optimizer.__call__()

        index: int = 0
        name: str = f"{optimizer}_{sensor}_{index}"
        saved_solutions: dict = self._read(Subobject.SOLUTION)
        while name in saved_solutions:
            index += 1
            name[-1] = str(index)

        variable_names: List[str] = list(sensor._trainable_variables_to_dict(optimizer.trainable_variables).keys())

        solution: dict = asdict(SolutionSave(
            variable_names,
            {var_name:optimizer._get_state_variable(StateVariable.TRAINABLE_VARIABLES).tolist()[i] for i, var_name in enumerate(variable_names)},
            optimizer._get_state_variable(StateVariable.GRADIENTS, all_epochs=True).tolist(),
            optimizer._get_state_variable(StateVariable.LOSS).tolist(),
            optimizer._get_state_variable(StateVariable.SENSITIVITY).tolist(),
            optimizer._get_state_variable(StateVariable.MEAN_SQUARED_ERROR).tolist(),
            optimizer._get_state_variable(StateVariable.FOOTPRINT).tolist(),
            optimizer._get_state_variable(StateVariable.SENSITIVITY_LOSS_WEIGHT).tolist(),
            optimizer._get_state_variable(StateVariable.MEAN_SQUARED_ERROR_LOSS_WEIGHT).tolist(),
            optimizer._get_state_variable(StateVariable.FOOTPRINT_LOSS_WEIGHT).tolist(),
            optimizer._get_state_variable(StateVariable.CONSTRAINT_PENALTY).tolist(),
            optimizer._get_state_variable(StateVariable.RESPONSE).tolist()
        ))

        self._save_object(Subobject.SOLUTION, name, solution)
        self._write(Subobject.SOLUTION, solution)
        pass

    def _display_(self, solution_name: str, display_vars: Tuple[str, ...]) -> None:
        fig, ax = plt.subplot()
        legend: list = []

        solution_save: dict = self.saved_objects[Subobject.SOLUTION][solution_name]
        for var in display_vars:
            match var:
                case 'GRA':
                    plt.plot(solution_save[StateVariable.GRADIENTS.name.lower()])
                    legend.append(StateVariable.GRADIENTS.name)
                case 'TRA':
                    data: List[List[float]] = solution_save[StateVariable.TRAINABLE_VARIABLES.name.lower()]
                    for i, name in enumerate(solution_save['variable_names']):
                        plt.plot(data[i])
                        legend.append(f"{StateVariable.TRAINABLE_VARIABLES.name}_{name}")
                case ''
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
            kw: Subobject = self._string_to_kw(args.which)
            self._list_(kw)
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
            self._delete_(self._string_to_kw(args.which), args.name)
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
            kw: Subobject = self._string_to_kw(args.which)
            if args.name is None:
                args.name = f"{kw.name}_{len(self.saved_objects[kw])+1}"
            self._configure_(kw, args.name)
        except KeyError as e:
            print(e)
        except argparse.ArgumentError as e:
            print(e)
        except SystemExit:
            pass
        pass
    
    def do_fit(self, arg):
        """
        Optimizes design specified in sensor object 
        <sensor>: Target sensor object
        <optimizer>: Optimizer
        """ 
        parser = argparse.ArgumentParser()
        parser.add_argument('sensor')
        parser.add_argument('optimizer')
        try:
            args = parser.parse_args()
            self._fit_(self, args.optimizer, args.sensor)
        except KeyError:
            print(f"Error: Sensor {args.sensor} does not exist")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_display(self, arg):
        pass