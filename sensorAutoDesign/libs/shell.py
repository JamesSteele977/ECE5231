import cmd, os, argparse
from .shellfn import ShellFn
from typing import IO

class UI(cmd.Cmd, ShellFn):
    def __init__(
            self,
            completekey: str = 'tab', 
            stdin: IO[str] | None = None, 
            stdout: IO[str] | None = None,
        ) -> None:
        cmd.Cmd.__init__(self, completekey, stdin, stdout)
        ShellFn.__init__(self)

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
        self._clear_()
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
        self._readme_()

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
            self._list_(self._string_to_kw(args.which))
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
            self._configure_(self._string_to_kw(args.which), args.name)
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
        <optimizer>: Target optimizer object
        """ 
        parser = argparse.ArgumentParser()
        parser.add_argument('sensor', help='Target sensor object')
        parser.add_argument('optimizer', help='Target optimizer object')
        try:
            args = parser.parse_args(arg.split())
            self._fit_(args.optimizer, args.sensor)
        except KeyError:
            print(f"Error: Sensor {args.sensor} and/or {args.optimizer} does not exist. Use list command to view existing objects or configure to create new.")
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass

    def do_display(self, arg):
        """
        Displays data from particular solve object 
        <solution>: Target solution object
        ["-s", "--sol"]: Inlcude flag to print final state of trainable variables
        ["-d", "--data"] <datum>: Tuple of data types to display        
        ["-a", "--all"]: Include flag to display all data. -d args ingnored if -a used
        ["-t", "--txt"]: Include flag to create txt file of either all (-a) data or selected (-d) data
        """ 
        parser = argparse.ArgumentParser()
        parser.add_argument('solution', type=str, help='Target solution object')
        parser.add_argument('-s', '--sol', action='store_true', help='Inlcude flag to print final state of trainable variables')
        parser.add_argument('-d', '--data', help='Tuple of data types to display')
        parser.add_argument('-a', '--all', action='store_true', help='Include flag to display all data. -d args ingnored if -a used')
        parser.add_argument('-t', '--txt', action='store_true', help='Include flag to create txt file of either all (-a) data or selected (-d) data')
        try:
            args = parser.parse_args(arg.split())
            self._display_(args.solution, args.sol, args.data, args.all, args.txt)
        except argparse.ArgumentError as e:
            print(str(e))
        except SystemExit:
            pass
        pass
