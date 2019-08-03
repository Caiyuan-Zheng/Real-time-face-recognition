# Real-time-face-recognition
using yolo-v3 and mobilefacenet to recognite faces and estimate age and gender

# reference  
https://github.com/yeziyang1992/Face_Recognition_Client  
https://github.com/sirius-ai/MobileFaceNet_TF\n  
https://github.com/ninesky110/Real-time-face-recognition  
https://github.com/gittigxuy/yolo-v3_face_detection

# environment
python==3.5.3  
tensorflow==1.9.0  
cuda==9.0  
single RTX 2080Ti  

# Detection
一开始使用的mtcnn模型做检测，但是由于mtcnn在检测人脸的过程中会做关键点检测，随着检测到的人脸数的增加，检测所需要的时间也会增加。
后来改用ssd模型，ssd模型速度足够快（10ms)，但是检测精度不高。
最后使用yolo-v3模型，精度相比ssd有了很大的提升，速度也在能接受的范围之内（在50ms左右）。

# Recognition
尝试过facenet和mobilefacenet，因为两个模型在lfw数据集都已经得到很高的正确率，于是我采用了速度快不少的mobilefacenet

#
