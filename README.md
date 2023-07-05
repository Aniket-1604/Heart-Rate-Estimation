# Heart-Rate-Estimation
This uses video of a person's face for finding his/her heart rate Firstly run the main python file by running "DPP.py" The command to be given is
DPP.py -f video.mp4
Then select a pixel on person's face to use as a key After saving the results it will show color amplified frames of the selected Region of Intrest The saved csv files has 4 .csv files with rgb values, relative rgb, yuv values and relative rgb values

Now to get HR from preprocessed csv file we use
hr.py to get HR from the rgb.csv file using green color as our main data
