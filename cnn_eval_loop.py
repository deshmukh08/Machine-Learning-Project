import cnn_eval as ce
import copy
import numpy as np

idArray = []
predictionArray = []
boolArray = []
zone = 14
for i in range(1,2): #(angle,angle+1)
	print('Begin Evaluating - ',i)
	apsid,predArray,bnotb = ce.evaluate(i, zone)
	
	idArray.append(copy.deepcopy(apsid))
	predictionArray.append(copy.deepcopy(predArray))
	boolArray.append(copy.deepcopy(bnotb))
	print('End Evaluating - ',i)

idArrayNP = np.array(idArray)
np.save('idArray',idArrayNP)

predictionArrayNP = np.array(predictionArray)
np.save('predictionArray',predictionArrayNP)

boolArrayNP = np.array(boolArray)
np.save('boolArray',boolArrayNP)

print('done')
