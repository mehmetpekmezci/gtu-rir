#!/usr/bin/env python3
from RIRHeader import *
from RIRLogger import *
from RIRData import *
from RIRDiffMethodEvaluator import *
from RIRReportGenerator import *




def main(data_dir,report_dir,selected_room_id):
  rirLogger=RIRLogger()
  print ("-1------------------------------------------------------------")
  rirData=RIRData(rirLogger.logger,data_dir,report_dir,selected_room_id)
  print ("-4------------------------------------------------------------")
  rirData.diffBetweenGeneratedAndRealRIRData()

  
  #rirDiffMethodEvaluator=RIRDiffMethodEvaluator(rirData)
  #rirDiffMethodEvaluator.compareRandomDataPointPairsInRealData()
  #rirDiffMethodEvaluator.compareRandomDataPointPairsInGeneratedData()
  #rirDiffMethodEvaluator.compareRandomDataPointPairsInRealDataSameRoom()
  #rirDiffMethodEvaluator.compareRandomDataPointPairsInGeneratedDataSameRoom()
  #rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameSpeakerPointInRealData()
  #rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameMicrophonePointInRealData()
  #rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameSpeakerPointInGeneratedData()
  #rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameMicrophonePointInGeneratedData()

  
  #rirReportGenerator=RIRReportGenerator(rirData)
  #rirReportGenerator.generateSummary()

  print("SCRIPT IS FINISHED")
  
                   

if __name__ == '__main__':
 main(str(sys.argv[1]).strip(),str(sys.argv[2]).strip(),str(sys.argv[3]).strip())



