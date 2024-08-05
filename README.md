This is almost a complete rewrite of https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html


It works fine on the example files given there, but I'm not sure if it works properly on my phone camera (with result on calibresult.jpg).
The chess pattern seems to be dependent on resolution of the image. And different scaling will give different intrinsic matrix, 
It doesn't seem like the image rescaling correlates linearly with intrinsic focal length.
I also need to research if phone autofocus affect the intrinsic properties.