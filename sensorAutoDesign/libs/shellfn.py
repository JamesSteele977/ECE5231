import json, os, sys, traceback
import matplotlib.pyplot as plt
from numpy import linspace
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .sensor import Sensor
from .optim import Optim
from .solution import Solution, StateVariable
from .config import SensorConfig, OptimConfig, SolutionSave, solutionSaveVariableNamesKey

from enum import Enum
from dataclasses import dataclass, asdict, fields, Field
from typing import Dict, List, Tuple

""" CONSTANTS """
descEntrySpaces: int = 24
hlineSpaceBuffer: int = 3
jsonDumpIndent: int = 4

nColorbarTicks: int = 10

windowsOSName: str = 'nt'
windowsClear: str = 'cls'
linuxMacClear: str = 'clear'

""" ENUMS """
class ShellPath(Enum):
    OUTPUT: str = './output'
    README: str = './README.md'
    TEMP: str = './cli/temp.json'
    SENSORS: str = './cli/sensors.json'
    OPTIMIZERS: str = './cli/optimizers.json'
    SOLUTIONS: str = './cli/solutions.json'

class Subobject(Enum):
    SENSOR: int = ShellPath.SENSORS.value
    OPTIMIZER: int = ShellPath.OPTIMIZERS.value
    SOLUTION: int = ShellPath.SOLUTIONS.value

""" CUSTOM TYPES """
configType: type = SensorConfig | OptimConfig | SolutionSave
pathType: type = ShellPath | Subobject

class ShellFn():
    """
    ShellFn provides foundational functions for the shell interface of the Sensor Design Optimization Tool,
    enabling the management of sensor, optimizer, and solution objects within the shell environment.

    It encapsulates methods for reading and writing JSON records, managing user interfaces, and handling
    tasks such as object configuration, deletion, listing, and displaying information. It serves as a base
    class for the UI shell, integrating various components of the tool.

    Attributes
    ----------
    saved_objects : Dict[Subobject, Dict[str, Sensor | Optim | Solution]]
        A dictionary storing instances of sensors, optimizers, and solutions, keyed by their respective names.

    Methods
    -------
    __init__(self):
        Initializes the shell function object, loading saved objects from JSON files into the shell environment.

    _read(self, kw: Subobject | ShellPath, temp_json=False) -> configType | dict:
        Reads and returns configuration data from specified JSON files.
        Parameters:
            kw : Subobject | ShellPath
                The type of object or the path to the JSON file.
            temp_json : bool, optional
                Flag to indicate if temporary JSON data is to be read.

    _write(self, kw: Subobject | ShellPath, data: dict):
        Writes data to a JSON file specified by the given keyword.
        Parameters:
            kw : Subobject | ShellPath
                The type of object or the path to the JSON file.
            data : dict
                Data to be written to the file.

    _list_(self, kw: Subobject):
        Lists all objects of a given type registered in the shell.
        Parameters:
            kw : Subobject
                The type of object to list.

    _delete_(self, kw: Subobject, name: str):
        Deletes a specified object saved in memory.
        Parameters:
            kw : Subobject
                The type of object to delete.
            name : str
                The name of the object to delete.

    _configure_(self, kw: Subobject, name: str):
        Configures an existing or creates a new object of the specified type.
        Parameters:
            kw : Subobject
                The type of object to configure or create.
            name : str
                The name of the object to configure or create.

    _fit_(self, optimizer_name: str, sensor_name: str):
        Optimizes the design specified in a sensor object using the designated optimizer.
        Parameters:
            optimizer_name : str
                The name of the optimizer object.
            sensor_name : str
                The name of the sensor object.

    _solution_(self, solution_name: str):
        Prints the final state of trainable variables for a given solution object.
        Parameters:
            solution_name : str
                The name of the solution object.

    _display_(self, solution_name: str, display_vars: Tuple[str, ...], display_all: bool, save_txt: bool):
        Displays and optionally saves data from a specified solution object.
        Parameters:
            solution_name : str
                The name of the solution object.
            display_vars : Tuple[str, ...]
                A tuple of data types to display.
            display_all : bool
                Flag to display all data; overrides 'display_vars' if set.
            save_txt : bool
                Flag to save the displayed data to a text file.
    """
    def __init__(self) -> None:
        self.saved_objects: Dict[Subobject, Dict[str, Sensor | Optim | Solution]] = {Subobject.SENSOR: {}, Subobject.OPTIMIZER: {}, Subobject.SOLUTION: {}}
        for subobject in Subobject: 
            saved: dict = self._read(subobject)
            for name, config in saved.items():
                self._save_object(subobject, name, config)
    
    #-----------------------------------------------------------------------------------------------
    """ MAIN USER FUNCTIONS """
    def _clear_(self) -> None:
        if os.name == windowsOSName:
            os.system(windowsClear)
            return
        os.system(linuxMacClear)

    def _readme_(self) -> None:
        if os.path.isfile(ShellPath.README.value):
            with open(ShellPath.README.value, 'r') as f:
                print(f.read())
            return
        print(f"{ShellPath.README} not found")
        
    def _list_(self, kw: Subobject) -> None:
        if kw == Subobject.SOLUTION:
            solution_list: str = '\n'.join([solution_name for solution_name in self.saved_objects[kw].keys()])
            print(solution_list)
            return
        self._print_description_header(kw)
        for name, entry in self.saved_objects[kw].items():
            self._print_description_line(name, entry)

    def _delete_(self, kw: Subobject, name: str) -> None:
        """
        Delete sensor/optimizer/solution object
        """    
        if not self._user_confirm(f"CONFIRM: Delete {kw.name} {name}?"):
            return
        
        name: str = name.upper()
        saved_objects: dict = self._read(kw)
        
        if name in saved_objects:
            del saved_objects[name]
        if name in self.saved_objects[kw]:
            del self.saved_objects[kw][name]
        
        self._write(kw, saved_objects)

    def _configure_(self, kw: Subobject, object_name: str | None = None) -> None:
        """
        Configure old or create new sensor/optimzer object
        """
        if kw == Subobject.SOLUTION:
            print(f"Cannot configure object type: Solution")
            return
        
        saved_objects: dict = self._read(kw)
        if object_name is None:
            object_name = f"{kw.name}_{len(self.saved_objects[kw])+1}"

        object_name: str = object_name.upper()
        if object_name in saved_objects:
            if not self._user_confirm(f"WARNING: {kw.name} {object_name} already exists. Overwrite?"):
                return

        config_format: configType = self._match_subobject_to_config_dataclass(kw)
        temp: dict = self._get_temp_json_dataclass_dict(config_format)
        while True:
            self._write(ShellPath.TEMP, temp)
            os.system(f"nano {ShellPath.TEMP.value}")
            try:
                with open(ShellPath.TEMP.value) as f:
                    user_config: dict = asdict(config_format.from_json(f.read()))
                break
            except KeyError as e:
                print(e)
            except TypeError as e:
                print(e)
    
        self._save_object(kw, object_name, user_config)
        self._write(kw, self.saved_objects[kw])
        self._print_description_header(kw)
        self._print_description_line(object_name, user_config)

    def _fit_(self, optimizer_name: str, sensor_name: str) -> None:
        sensor_obj: Sensor = Sensor(SensorConfig(**self.saved_objects[Subobject.SENSOR][sensor_name.upper()]))
        optimizer_obj: Optim = Optim(OptimConfig(**self.saved_objects[Subobject.OPTIMIZER][optimizer_name.upper()]), sensor_obj.sensor_profile)
        optimizer_obj.__call__()

        index: int = 0
        name: str = f"{optimizer_name}_{sensor_name}_{index}"
        saved_solutions: dict = self._read(Subobject.SOLUTION)
        while name.upper() in saved_solutions:
            index += 1
            name = name[:-1]+str(index)

        self._save_object(Subobject.SOLUTION, name.upper(), self._get_writable_solution(sensor_obj, optimizer_obj))
        self._write(Subobject.SOLUTION, self.saved_objects[Subobject.SOLUTION])

    # _fit_ sub
    def _get_writable_solution(self, sensor_obj: Sensor, optimizer_obj: Optim) -> dict:
        variable_names: List[str] = list(sensor_obj._trainable_variables_to_dict(optimizer_obj.trainable_variables).keys())
        if len(variable_names) == 1:
            trainable_variables_save_entry: Dict[str, List[float]] = {variable_names[0]:optimizer_obj._get_state_variable(StateVariable.TRAINABLE_VARIABLES, all_epochs=True).tolist()}
        else:
            trainable_variables_save_entry: Dict[str, List[float]] = {var_name:optimizer_obj._get_state_variable(StateVariable.TRAINABLE_VARIABLES, all_epochs=True).tolist()[i] for i, var_name in enumerate(variable_names)}
        state_variables_less_tvars: List[StateVariable] = list(StateVariable)
        state_variables_less_tvars.remove(StateVariable.TRAINABLE_VARIABLES)
        solution_save_config: List[List[float]] = [optimizer_obj._get_state_variable(state_variable, all_epochs=True).tolist() for state_variable in state_variables_less_tvars]
        solution_save_config[0:0]: list = [variable_names, trainable_variables_save_entry]
        return asdict(SolutionSave(*solution_save_config)) 
    #---

    def _display_(self, solution_name: str, print_final_epoch_solution: bool, queried_vars: Tuple[str, ...] | None, display_all: bool, save_txt: bool) -> None:
        if (queried_vars is None) and (not display_all):
            print("WARNING: Invalid query. User must provide either --data arguments or --all flag.")
            return
        if print_final_epoch_solution:
            self._print_final_epoch_solution(solution_name)
        if display_all:
            queried_vars: tuple = tuple([statevar.name for statevar in StateVariable])

        solution_save: dict = self.saved_objects[Subobject.SOLUTION][solution_name.upper()]
        self._plot_queried_data(queried_vars, solution_save)

        if save_txt:
            self._save_solution_to_txt(solution_name, solution_save)

    # _display_ sub
    def _print_final_epoch_solution(self, solution_name: str):
        print(self._recursively_format_obj_to_str({
            variable_name:values[-1] for variable_name, values in self.saved_objects[
                Subobject.SOLUTION
            ][
                solution_name.upper()
            ][
                StateVariable.TRAINABLE_VARIABLES.name.lower()
            ].items()
        }))
    
    def _plot_queried_data(self, queried_vars: Tuple[str, ...], solution_save: dict) -> None:
        num_plots: int = 1
        if StateVariable.RESPONSE.name in [var.upper() for var in queried_vars]:
            num_plots: int = 2
        main_ax = plt.subplot(num_plots, 1, 1)
        legend: list = []

        for queried_var in queried_vars:
            if queried_var.lower() in solution_save:
                match queried_var.upper():
                    case StateVariable.RESPONSE.name:
                        self._plot_response(solution_save[StateVariable.RESPONSE.name.lower()])
                    case StateVariable.TRAINABLE_VARIABLES.name:
                        main_ax, legend = self._plot_trainable_variables(
                            main_ax, legend, 
                            solution_save[StateVariable.TRAINABLE_VARIABLES.name.lower()], 
                            solution_save[solutionSaveVariableNamesKey]
                        )
                    case _:
                        main_ax, legend = self._plot_main(main_ax, legend, queried_var.upper(), solution_save[queried_var.lower()])

        main_ax.legend(legend)
        plt.tight_layout()
        plt.show()

    #_plot_queried_data sub
    def _plot_trainable_variables(self, main_ax, legend: list, trainable_variable_dict: Dict[str, List[float]], trainable_variable_names: List[str]) -> tuple:
        for trainable_variable_name in trainable_variable_names:
            main_ax, legend = self._plot_main(main_ax, legend, trainable_variable_name, trainable_variable_dict[trainable_variable_name])
        return main_ax, legend

    def _plot_response(self, response_data: List[List[float]]) -> None:
        response_ax = plt.subplot(2, 1, 2)
        epochs: int = len(response_data)
        cmap = plt.cm.inferno
        colors = cmap(linspace(0, 1, epochs))

        for i in range(epochs):
            response_ax.plot(response_data[i], color=colors[i])

        plt.colorbar(ScalarMappable(Normalize(1, epochs), cmap), ax=response_ax, ticks=linspace(1, epochs, nColorbarTicks))
    
    def _plot_main(self, main_ax, legend: list, data_label: str, data: list) -> tuple:
        main_ax.plot(data)
        legend.append(data_label)
        return main_ax, legend
    #---

    def _save_solution_to_txt(self, solution_name: str, solution_save: dict) -> None:
        filename: str = os.path.join(ShellPath.OUTPUT.value, f"{solution_name}.txt")
        with open(filename, 'w') as f:
            for key, value in solution_save.items():
                f.write(f"{key}\n{value}\n")
        print(f"Output saved to {filename}")
    #---

    #-----------------------------------------------------------------------------------------------
    """ REFACTORING """
    ### OBJECT RECORDS
    # read/write json files
    def _read(self, kw: Subobject | ShellPath) -> Tuple[configType, ...]:
        self._check_rectify_path(kw.value)
        try:
            with open(kw.value, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Incorrect formatting during read file {kw.value}")
        except KeyError as e:
            print(f"KeyError: {e}")
        except TypeError as e:
            print(f"TypeError: {e}")

    def _write(self, kw: Subobject | ShellPath, data: dict) -> None:
        self._check_rectify_path(kw.value)
        with open(kw.value, 'w') as f:
            json.dump(data, f, indent=jsonDumpIndent)

    # _read, _write sub
    def _check_rectify_path(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            split = os.path.join(*os.path.split(filepath)[:-1])
            if not os.path.exists(split):
                os.mkdir(split)
        elif not os.path.isfile(filepath):
            os.rmdir(filepath)
        else: # If path is valid
            return
        with open(filepath, 'w') as f: # Initialize json dict structure
            json.dump({}, f)
    #---

    # write/delete from saved_objects attr
    def _save_object(self, kw: Subobject, name: str, config: dict, verbose: bool = True) -> None:
        self.saved_objects[kw][name]: dict = config

    ### SUBOBJECT MAPPINGS
    # subobject -> configuration dataclass
    def _match_subobject_to_config_dataclass(self, subobject: Subobject) -> configType | int:
        match subobject:
            case Subobject.SENSOR:
                return SensorConfig
            case Subobject.OPTIMIZER:
                return OptimConfig
            case Subobject.SOLUTION:
                return SolutionSave

    # UI string -> subobject
    def _string_to_kw(self, string: str) -> Subobject:
        match string[:3].upper():
            case 'SOL':
                return Subobject.SOLUTION
            case 'OPT':
                return Subobject.OPTIMIZER
            case 'SEN':
                return Subobject.SENSOR
        raise KeyError(f"Invalid object type {string}. Choose from: 'Solution', 'Optimizer', 'Sensor'")

    ### USER INPUT
    def _user_confirm(self, msg: str) -> bool:
        confirm: str = str(input(f"{msg} [y/n] ")).upper()
        while confirm not in {'Y', 'N'}:
            confirm: str = str(input(f"Invalid response '{confirm}' [y/n] ")).upper()
        if confirm == 'N':
            print('Exiting process...')
            return False
        return True

    ### DATACLASS MANIPULATION
    def _get_temp_json_dataclass_dict(self, dataclass: dataclass) -> dict:
        dataclass_dict: dict = {}
        for field in fields(dataclass):
            match field.type.__name__.lower():
                case str.__name__ | float.__name__ | int.__name__:
                    val: str = str()
                case dict.__name__:
                    val: dict = dict()
                case tuple.__name__ | list.__name__ | set.__name__:
                    val: list = list()
                case _:
                    print('WARNING: unknown type enountered in config')
            dataclass_dict[field.name] = val
        return dataclass_dict
    
    ### OBJECT UI DESCRIPTION
    # table printing
    def _print_description_line(self, name: str, config: dict) -> None:
        print(f"{name[:descEntrySpaces].ljust(descEntrySpaces)}|{'|'.join([self._recursively_format_obj_to_str(label)[:descEntrySpaces].ljust(descEntrySpaces) for label in config.values()])}")
        
    def _print_description_header(self, kw: Subobject) -> None:
        if len(self.saved_objects[kw]) != 0:
            config: dataclass = self._match_subobject_to_config_dataclass(kw)
            print(f"{'Name'[:descEntrySpaces].ljust(descEntrySpaces)}|{'|'.join([field.name[:descEntrySpaces].ljust(descEntrySpaces) for field in fields(config)])}\n{'-'*(len(fields(config)) * (hlineSpaceBuffer + descEntrySpaces))}")
        else:
            print("< none >")
    
    # print formatting
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
            
    ### ERROR HANDLING
    def _general_exception_(self) -> None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception type:", exc_type)
        print("Exception value:", exc_value)
        traceback_details = traceback.format_tb(exc_traceback)
        print("Detailed traceback:")
        for line in traceback_details:
            print(line)