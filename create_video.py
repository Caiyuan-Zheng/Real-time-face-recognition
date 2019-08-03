import skvideo.io
import skvideo.datasets
import numpy as np 
import cv2
 
# write an ndarray to a video file
outputdata = np.random.random(size=(30,480,680,3)) * 255
outputdata = outputdata.astype(np.uint8)
skvideo.io.vwrite("outputvideo.mp4", outputdata)
 
# FFmpeg writer (报错)
outputdats = np.random.random(size=(100,480,680,3)) * 255
outputdata = outputdata.astype(np.uint8)
fps=10
writer = skvideo.io.FFmpegWriter("outputvideoplus.mp4", inputdict={'-r': fps, '-width': 720, '-height': 480},
            outputdict={'-r': fps, '-vcodec': 'libx264', '-pix_fmt': 'h264'}
)
image=cv2.imread("tmp.jpg")
image=np.array(image)
for i in range(100):
    writer.writeFrame(image)
writer.close()