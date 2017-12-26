import os
angles = [1, 13, 9]
zone = 6
for i in angles:
    run_str = "python cnn.py "+str(i)+" "+str(zone)
    print("Start: "+run_str)
    os.system(run_str)
    print("End: "+run_str)

