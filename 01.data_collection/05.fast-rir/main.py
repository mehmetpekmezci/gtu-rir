#!/usr/bin/env python3
from RIRHeader import *
from RIRLogger import *
from RIRData import *
from RIRDiffMethodEvaluator import *
from RIRReportGenerator import *




def main(fast_rir_dir):
  rirLogger=RIRLogger()
  print ("-1------------------------------------------------------------")
  rirData=RIRData(rirLogger.logger,fast_rir_dir)
  print ("-2------------------------------------------------------------")
  rirData.generateFastRIRInputs()
  print ("-3------------------------------------------------------------")
  rirData.triggerFastRIRToGenerateRIRsAndT60s()
  print ("-4------------------------------------------------------------")
  rirData.diffBetweenGeneratedAndRealRIRData()

  
  rirDiffMethodEvaluator=RIRDiffMethodEvaluator(rirData)
  rirDiffMethodEvaluator.compareRandomDataPointPairsInRealData()
  rirDiffMethodEvaluator.compareRandomDataPointPairsInGeneratedData()
  rirDiffMethodEvaluator.compareRandomDataPointPairsInRealDataSameRoom()
  rirDiffMethodEvaluator.compareRandomDataPointPairsInGeneratedDataSameRoom()
  rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameSpeakerPointInRealData()
  rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameMicrophonePointInRealData()
  rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameSpeakerPointInGeneratedData()
  rirDiffMethodEvaluator.compareDataPointPairsThatHasTheSameMicrophonePointInGeneratedData()

  
  rirReportGenerator=RIRReportGenerator(rirData)
  rirReportGenerator.generateSummary()

  
                   

if __name__ == '__main__':
 main(str(sys.argv[1]).strip())



