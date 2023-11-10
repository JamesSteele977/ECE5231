import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from libs.shell import UI

frontend = UI()
frontend.cmdloop()
