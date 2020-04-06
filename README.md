# HCF_Tracker in Python
 
&nbsp;Algorithms Name: Hierarchical Convolutional Features for Visual Tracking </br>
&nbsp;Author : Kevin Ke </br>
&nbsp;Date : 28th, March, 2020 </br>


### Requirements
- Python 3.5
- NumPy
- OpenCV3
- pytorch 1.2
- cuda10.0.1

### Use
Download the git and
```shell
git clone https://github.com/kevin655/HCF_Tracker.git
cd HCF
python main.py
```
It will open the default camera of your computer, and the groundtruth for the first frame can be set in ./Truth.txt of shape (x1, y1, x2, y2).

### Test on your own video
For time reasons, I didn't add ArgumentParser. But you can edit my code in "main.py" by changing the "get_video" function. 
Also remember to change the groundtruth in ./Truth.txt.
