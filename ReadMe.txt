The Model has mainly three parts.
1. Data pre-processing
segment.py, csv_processing.py takes care the data processing parts.
csv_processing.py is individual runnable python file while segment.py is used alog with other functions.

2. Training the Model.
cnn.py
usage: python cnn.py <angle> <zone>
cnn_loop.py allows you to train the model at a  time for multiple angles.
open the file and mention the anlgles in the angles array. eg. angles = [1, 9, 13]

3. Testing the model
cnn_eval_loop.py change the zone and angle you want to test the data for and runn the cnn_eval_loop.py
this prints the threat detetcted images for a particuar zone. and reports the True Posatives, False Postaives, True Negatives and False Negatives.

All the py files are listed in the home directory
src code - home dirctore of projtec
All Tfl - models are listed in models dir of home dir
All csv flies are listed in - csvFiles folder of the home dir
All aps files are listed in the aps directory, which is parallel to the home directory.
