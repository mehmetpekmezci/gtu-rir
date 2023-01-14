#!/usr/bin/env python3
from RIRHeader import *

class RIRLogger :
 def __init__(self):
     self.script_dir=os.path.dirname(os.path.realpath(__file__))
     self.script_name=os.path.basename(self.script_dir)
     self.log_dir_for_logger=self.script_dir+"/../../../../logs/logger/"+self.script_name

     #if not os.path.exists(self.log_dir_for_tfsummary):
     #    os.makedirs(self.log_dir_for_tfsummary)
     if not os.path.exists(self.log_dir_for_logger):
         os.makedirs(self.log_dir_for_logger)
    
     ## CONFUGRE LOGGING
     self.logger=logging.getLogger('rir')
     
     self.logger.propagate=False
     self.logger.setLevel(logging.INFO)

#     loggingFileHandler = logging.FileHandler(self.log_dir_for_logger+'/rir-'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))+'.log')
#     loggingFileHandler.setLevel(logging.DEBUG)
     loggingConsoleHandler = logging.StreamHandler()
     loggingConsoleHandler.setLevel(logging.DEBUG)
     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     loggingFileHandler.setFormatter(formatter)
     loggingConsoleHandler.setFormatter(formatter)
#     self.logger.addHandler(loggingFileHandler)
     self.logger.addHandler(loggingConsoleHandler)



