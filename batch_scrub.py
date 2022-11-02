import os
startYear = 1895
endYear = 2019

for i in range(startYear,endYear+1):
    os.system('python scrubber.py '+str(i))