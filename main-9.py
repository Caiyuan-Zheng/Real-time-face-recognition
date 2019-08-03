<<<<<<< HEAD
import sys
import time
import cv2
import os
import shutil
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
import sys
sys.path.append("../")
import tensorflow as tf
import math
import pickle
import PIL.Image as Image
import facenet
from scipy import misc
from sklearn.svm import SVC
# import helpers
# 界面布局
from gui import Ui_widget
from yolo_detection import YOLO
from age_gender_classify import AgeGenderClassfier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import Augmentor
import shutil


def augmentation(input_path,output_path):
    path="align_database"
    if os.path.exists(output_path):
        shutil.rmtree(output_path) 
    os.mkdir(output_path)
    files=os.listdir(input_path)
    for person in files:
        file=os.path.join(input_path,person)
        p=Augmentor.Pipeline(file)
        p.rotate(probability=0.5,max_left_rotation=2,max_right_rotation=2)
        p.skew_tilt(probability=0.5,magnitude=0.02)#上下左右方向的垂直型变，参数magnitude为型变的程度（0，1
        p.skew_corner(probability=0.5,magnitude=0.02)#向四个角形变
        p.random_distortion(probability=0.5,grid_height=5,grid_width=5,magnitude=1)#弹性扭曲
        p.shear(probability=0.5,max_shear_left=2,max_shear_right=2)#使图像向某一侧倾斜啦,参数与旋转类似，范围是0-25
        p.flip_left_right(probability=0.2)
        #p.random_erasing(probability=0.5,rectangle_area=0.15)#这个函数是随机遮盖掉图像中的某一个部分，rectangle_area的变化范围为0.1-1
        p.sample(500)
        tmp_path=os.path.join(file,"output")
        new_path=os.path.join(output_path,person)
        shutil.move(tmp_path,new_path)


def load_detection_model(detection_graph,detection_model_path):
     with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(detection_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')   


#图片预处理阶段
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  
def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def load_image(image_old, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    if image_old.ndim == 2:
        image_old = to_rgb(image_old)
    if do_prewhiten:
        image_old = prewhiten(image_old)
    image_old = crop(image_old, do_random_crop, image_size)
    image_old = flip(image_old, do_random_flip)
    return image_old

def load_recognition_model(recognition_graph,recognition_sess,model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    with recognition_graph.as_default():
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with tf.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(recognition_sess, os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file



class MyDesignerShow(QtWidgets.QWidget, Ui_widget):#继承了Ui_widget的属性
    _signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super(MyDesignerShow, self).__init__()
        self.timer_camera = QtCore.QTimer()   # 本地摄像头定时器
        self.timer_video = QtCore.QTimer()  # video定时器
        self.cap = None     
        self.video_output=None             
        self.CAM_NUM=0  #摄像头编码
        self.add_face=False
        self.add_path=""
        self.add_face_begin=time.time()
        self.add_face_end=time.time()
        # 获取摄像头编号
        self.add_picture_path="align_database"
        self.train_picture_path="augmention"
        self.facenet_model_path="face_models/20180408-102900"
        self.mobilenet_model_path="face_models/MobileFaceNet_9925_9680.pb"
        self.database_path="Database.npz"
        self.SVCpath="face_models/SVCmodel.pkl"
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.ssd_model_path = 'face_models/frozen_inference_graph_face.pb'
        self.yolo_model_path=""
        self.age_gender_model_path="models"
        self.recognition_graph=tf.Graph()
        self.recognition_sess=tf.Session(graph=self.recognition_graph)
        self.count=0
        self.age_count={}
        self.age_sum={}

        # Load the model
        print('Loading feature extraction model')
        with open(self.SVCpath, 'rb') as infile:
            (self.classifymodel, self.class_names) = pickle.load(infile)
        print('Loaded classifier model from file "%s"' % self.SVCpath)

        # Get input and output tensors
        '''
        facenet.load_model(self.sess,self.facenet_model_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        '''
        load_recognition_model(self.recognition_graph,self.recognition_sess,self.mobilenet_model_path)
        self.images_placeholder=self.recognition_graph.get_tensor_by_name("input:0")
        self.embeddings=self.recognition_graph.get_tensor_by_name("embeddings:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        self.Database=np.load(self.database_path)
        self.detector=YOLO(self.yolo_model_path)
        self.age_gender_classifier=AgeGenderClassfier(self.age_gender_model_path)

        self.setupUi(self)                          # 加载窗体
        #以下是将按钮和功能联系起来
        self.btn_close.clicked.connect(self.close)   # 关闭程序
        self.btn_local_camera.clicked.connect(self.get_local_camera)   # 打开本地相机
        self.btn_from_video.clicked.connect(self.get_from_video)

        self.btn_get_faces.clicked.connect(self.get_faces)              # 得到人脸图像
        self.btn_delete_face.clicked.connect(self.delete_faces)                    # 报错
        self.btn_train_classifier.clicked.connect(self.train_classifier)                # 新建人脸数据

        self.timer_camera.timeout.connect(self.openFrame)  # 计时结CAM_NUM束调用open_frame方法
        self.timer_video.timeout.connect(self.openFrame)    #计时结束调用open_frame方法
        self.time_start=time.time()
        self.time_end=time.time()

    def openFrame(self):
        self.time_end=time.time()
        print ("allt time",self.time_end-self.time_start)
        t=self.time_end-self.time_start
        fps=cv2.getTickFrequency()/t
        self.time_start=time.time()
        t0=time.time()
        ret,frame = self.cap.read()
        if(self.cap.isOpened()):
            ts = cv2.getTickCount()
            ret, frame = self.cap.read()
            if ret:
                t1=time.time()
                height=frame.shape[0]
                width=frame.shape[1]
                image=np.array(frame)
                t2=time.time()
                boxes_c=self.detector.detect_image(frame)
                t3=time.time()
                imgs=[]
                ag_imgs=[]
                if boxes_c.shape[0]>0:
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]#检测出的人脸区域，左上x，左上y，右下x，右下y
                        score = boxes_c[i, 4]#检测出人脸区域的得分
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        margin=int((bbox[3]-bbox[1])*0.25)
                        x1=np.maximum(int(bbox[0]-margin),0)
                        y1=np.maximum(int(bbox[1]-margin),0)
                        x2=np.minimum( int(bbox[2]+margin),width)
                        y2=np.minimum( int(bbox[3]+margin),height)
                        crop_img=image[y1:y2,x1:x2]

                        ag_margin=int((bbox[3]-bbox[1])*0.4)
                        ag_x1=np.maximum(int(bbox[0]-ag_margin),0)
                        ag_y1=np.maximum(int(bbox[1]-ag_margin),0)
                        ag_x2=np.minimum( int(bbox[2]+ag_margin),width)
                        ag_y2=np.minimum( int(bbox[3]+ag_margin),height)
                        ag_crop_img=image[y1:y2,x1:x2]

                        #cv2.imwrite("tmp.png",crop_img)
                        scaled=misc.imresize(crop_img,(112,112),interp='bilinear')
                        img=load_image(scaled,False, False,112)
                        self.count+=1                       
                        imgs.append(img)
                        ag_img=misc.imresize(ag_crop_img,(160,160),interp='bilinear')
                        ag_img=load_image(ag_img,False, False,160)
                        ag_imgs.append(ag_img)

                        #当添加人脸时，把检测到的人脸保存在文件夹下
                        if self.add_face:
                            print ("添加人脸剩余时间",self.add_face_end-time.time())
                            output=misc.imresize(crop_img,(112,112))
                            cv2.imwrite(self.add_path+"/"+"%d.png" %(int(time.time()*100)),output)
                            #cv2.imwrite("train_output/"+"%d.png" %(int(time.time()*100)),output)
                            if time.time()>self.add_face_end:#停止添加人脸
                                self.timer_camera.stop()
                                self.label_camera.clear()
                                self.add_face=False
                                self.count=0
                                self.video_output.release()
                                self.btn_get_faces.setText("添加人脸数据")
                    #下面进行人脸识别，年龄检测，性别检测
                    if not self.add_face:
                        feed_dict = { self.images_placeholder:imgs}
                        embvecor=self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
                        embvecor=np.array(embvecor)

                        #利用SVM对人脸特征进行分类
                        predictions = self.classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lables=self.class_names[best_class_indices]
                        #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        face_num=boxes_c.shape[0]
                        for i in range(face_num):
                            best_class_probability=predictions[i][best_class_indices[i]]
                            print(best_class_probability)
                            if best_class_probability<0.4:
                                tmp_lables[i]="others"

                        ages,genders=self.age_gender_classifier.classify(ag_imgs)
                        for i in range(face_num):
                            if tmp_lables[i] not in self.age_count.keys():
                                self.age_sum[tmp_lables[i]]=ages[i]
                                self.age_count[tmp_lables[i]]=1
                            else:
                                self.age_sum[tmp_lables[i]]+=ages[i]
                                self.age_count[tmp_lables[i]]+=1                               
                            bbox = boxes_c[i, :4]#检测出的人脸区域，左上x，左上y，右下x，右下y
                            score = boxes_c[i, 4]#检测出人脸区域的得分
                            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                            gender="F" if genders[i] == 0 else "M"
                            age=int(self.age_sum[tmp_lables[i]]/self.age_count[tmp_lables[i]])
                            cv2.putText(frame, '{},{},{}'.format(tmp_lables[i],gender,int(age)), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)                                

                t4=time.time()
                cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255), 2)
                output=frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data,  width, height, bytesPerLine, 
                                 QtGui.QImage.Format_RGB888).scaled(self.label_camera.width(), self.label_camera.height())
                self.video_output.write(output)
                #cv2.imwrite("tmp.jpg",frame)
                #self.label_camera.setPixmap(QtGui.QPixmap.fromImage(q_image)) 
                t5=time.time()
                print ("before detect",t2-t1,"detect",t3-t2,"recognite",t4-t3,"show",t5-t4,"total",t5-t1)
            else:
                self.cap.release()
                self.video_output.release()
                self.timer_camera.stop()   # 停止计时器
                self.timer_video.stop()
                self.btn_from_video.setText(u'打开视频')

    # 获取本地摄像头视频
    def get_local_camera(self):
        if self.timer_video.isActive() or self.add_face:      # 查询网络摄像头是否打开,或者是否正在添加人脸
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请先关闭视频或者等待添加人脸完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        elif not self.timer_camera.isActive():
            self.cap=cv2.VideoCapture(self.CAM_NUM)

            flag = self.cap.isOpened()
            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.age_count={}
                self.age_sum={}
                size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print ("video fps",video_fps)
                self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), 12.5, size,1)
                self.timer_camera.start(0)     # 30ms刷新一次
                self.btn_local_camera.setText(u'关闭本地摄像头')

        else:
            self.timer_camera.stop()    # 定时器关闭
            self.cap.release()          # 摄像头释放
            self.label_camera.clear()   # 视频显示区域清屏RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
            self.video_output.release()
            #self.graphicsView.show()
            self.btn_local_camera.setText(u'打开本地摄像头')

    def get_from_video(self):
        if self.timer_camera.isActive() or self.add_face:#查询video是否打开
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请先关闭摄像头或者等待添加人脸完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        elif not self.timer_video.isActive():                      
            videopath,ok= QtWidgets.QFileDialog.getOpenFileName(self,"getOpenFileName","video") 
            if ok:
                self.age_count={}
                self.age_sum={}
                print (videopath)
                self.cap=cv2.VideoCapture(videopath)

                size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), video_fps, size,1)
                if self.cap is None:
                    QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测视频格式是否正确", buttons=QtWidgets.QMessageBox.Ok,
                                    defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_video.start(0)
                    self.btn_from_video.setText(u"关闭视频")
        else:
            self.timer_video.stop()
            self.cap.release()
            #self.video_output.release()
            self.label_camera.clear()
            self.btn_from_video.setText(u'打开视频')



    def get_faces(self):
        name,add_ok=QInputDialog.getText(self,"请输入要添加的人脸姓名","",QLineEdit.Normal,"")
        if add_ok:
            self.add_path=os.path.join(self.add_picture_path,name)
            if os.path.exists(self.add_path):
                save_ok=QtWidgets.QMessageBox.warning(self, u"Warning", u"人脸库中已经包含该姓名，是否删除原人脸", buttons=QtWidgets.QMessageBox.Save| QMessageBox.Discard | QMessageBox.Cancel,
                                    defaultButton=QtWidgets.QMessageBox.Save)
                if save_ok==QMessageBox.Cancel:
                    print ("取消添加人脸")
                    return
                if save_ok==QMessageBox.Discard:
                    print ("放弃保存之前的人脸")
                    shutil.rmtree(self.add_path)
                    os.mkdir(self.add_path)
            else:
                os.mkdir(self.add_path)
            #flag = self.cap.open(self.CAM_NUM)
            #self.cap=cv2.VideoCapture("/home/zhengcy/Real-time-face-recognition-master/test/video_test.mp4")
            self.cap=cv2.VideoCapture(self.CAM_NUM)
            size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), video_fps, size,1)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.add_face=True
                self.btn_get_faces.setText("正在添加人脸")
                self.add_face_begin=time.time()+3
                while(time.time()<self.add_face_begin):
                    time.sleep(0.5)
                    #cv2.putText(frame, "人脸录制将在 " +"{:.3f}".format(time_begin-time.time())+"之后开始", (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    #    (255, 0, 255), 2)
                    self.label_camera.setText("人脸录制将在 " +"{:.3f}".format(self.add_face_begin-time.time())+"之后开始")
                self.timer_camera.start(0)
                self.add_face_end=self.add_face_begin+15

    def delete_faces(self):
        name,add_ok=QInputDialog.getText(self,"请输入要删除的人脸姓名","请输入要删除的人脸姓名",QLineEdit.Normal,"")
        if add_ok:
            delete_path=os.path.join(self.add_picture_path,name)
            if not os.path.isdir(delete_path):
                QtWidgets.QMessageBox.warning(self, u"Warning", u"人脸库中不存在该姓名", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                shutil.rmtree(delete_path) 
                QtWidgets.QMessageBox.warning(self, u"Warning", u"成功删除人脸", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)     



    def debug(self):
        self.face_recog.reload_data()  # 重载人脸数据集
        num = self.face_recog.max_num
        file_names = self.face_recog.names
        print(file_names)
        if num > 0:
            result, ok = QtWidgets.QInputDialog.getItem(self, u"人脸数据校验", u"把人脸数据存入对应的文件夹中，可增加人脸识别的准确性。确定把图片存放在以下文件夹中吗？",
                                                        file_names, 1, False)
            if ok:
                if self.face_photo is not None:
                    # 保存图片
                    s_time = time.ctime().replace(' ', '_').replace(':', '_')
                    cv2.imwrite('./faces/' + result + '/' + str(s_time) + '.jpg', self.face_photo)
                    self.textEdit.append("已保存在./faces/" + result + '文件夹下!!')
        else:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"数据集为空,请新建人脸数据！", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    def train_classifier(self):
        augmentation(self.add_picture_path,self.train_picture_path)
        self.face2database()
        print("Train Classifier")
        #database_path为人脸数据库
        #SVCpath为分类器储存的位置
        Database=np.load(self.database_path)
        name_lables=Database['lab']
        embeddings=Database['emb']
        name_unique=np.unique(name_lables)
        labels=[]
        for i in range(len(name_lables)):
            for j in range(len(name_unique)):
                if name_lables[i]==name_unique[j]:
                    labels.append(j)
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(embeddings, labels)
        with open(self.SVCpath, 'wb') as outfile:
            pickle.dump((model,name_unique), outfile)
            print('Saved classifier model to file "%s"' % self.SVCpath)
        reply = QtWidgets.QMessageBox.information(self,
                                    "information",  
                                    "分类器训练完成。",  
                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def face2database(self,batch_size=90,image_size=112):
        #提取特征到数据库
        #picture_path为人脸文件夹的所在路径
        #model_path为facenet模型路径
        #database_path为人脸数据库路径
        dataset = facenet.get_dataset(self.train_picture_path)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False,image_size)
            feed_dict = { self.images_placeholder:images}
            emb_array[start_index:end_index,:] = self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
        np.savez(self.database_path,emb=emb_array,lab=labels)
        print("数据库特征提取完毕！")
        #emb_array里存放的是图片特征，labels为对应的标签


if __name__ == "__main__":
    if not os.path.exists("./faces"):
        os.makedirs("./faces")
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyDesignerShow()    # 创建实例
    myshow.show()           # 使用Qidget的show()方法
    sys.exit(app.exec_())

=======
import sys
import time
import cv2
import os
import shutil
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
import sys
sys.path.append("../")
import tensorflow as tf
import math
import pickle
import PIL.Image as Image
import facenet
from scipy import misc
from sklearn.svm import SVC
# import helpers
# 界面布局
from gui import Ui_widget
from yolo_detection import YOLO
from age_gender_classify import AgeGenderClassfier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import Augmentor
import shutil


def augmentation(input_path,output_path):
    path="align_database"
    if os.path.exists(output_path):
        shutil.rmtree(output_path) 
    os.mkdir(output_path)
    files=os.listdir(input_path)
    for person in files:
        file=os.path.join(input_path,person)
        p=Augmentor.Pipeline(file)
        p.rotate(probability=0.5,max_left_rotation=2,max_right_rotation=2)
        p.skew_tilt(probability=0.5,magnitude=0.02)#上下左右方向的垂直型变，参数magnitude为型变的程度（0，1
        p.skew_corner(probability=0.5,magnitude=0.02)#向四个角形变
        p.random_distortion(probability=0.5,grid_height=5,grid_width=5,magnitude=1)#弹性扭曲
        p.shear(probability=0.5,max_shear_left=2,max_shear_right=2)#使图像向某一侧倾斜啦,参数与旋转类似，范围是0-25
        p.flip_left_right(probability=0.2)
        #p.random_erasing(probability=0.5,rectangle_area=0.15)#这个函数是随机遮盖掉图像中的某一个部分，rectangle_area的变化范围为0.1-1
        p.sample(500)
        tmp_path=os.path.join(file,"output")
        new_path=os.path.join(output_path,person)
        shutil.move(tmp_path,new_path)


def load_detection_model(detection_graph,detection_model_path):
     with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(detection_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')   


#图片预处理阶段
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  
def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def load_image(image_old, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    if image_old.ndim == 2:
        image_old = to_rgb(image_old)
    if do_prewhiten:
        image_old = prewhiten(image_old)
    image_old = crop(image_old, do_random_crop, image_size)
    image_old = flip(image_old, do_random_flip)
    return image_old

def load_recognition_model(recognition_graph,recognition_sess,model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    with recognition_graph.as_default():
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with tf.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(recognition_sess, os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file



class MyDesignerShow(QtWidgets.QWidget, Ui_widget):#继承了Ui_widget的属性
    _signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super(MyDesignerShow, self).__init__()
        self.timer_camera = QtCore.QTimer()   # 本地摄像头定时器
        self.timer_video = QtCore.QTimer()  # video定时器
        self.cap = None     
        self.video_output=None             
        self.CAM_NUM=0  #摄像头编码
        self.add_face=False
        self.add_path=""
        self.add_face_begin=time.time()
        self.add_face_end=time.time()
        # 获取摄像头编号
        self.add_picture_path="align_database"
        self.train_picture_path="augmention"
        self.facenet_model_path="face_models/20180408-102900"
        self.mobilenet_model_path="face_models/MobileFaceNet_9925_9680.pb"
        self.database_path="Database.npz"
        self.SVCpath="face_models/SVCmodel.pkl"
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.ssd_model_path = 'face_models/frozen_inference_graph_face.pb'
        self.yolo_model_path=""
        self.age_gender_model_path="models"
        self.recognition_graph=tf.Graph()
        self.recognition_sess=tf.Session(graph=self.recognition_graph)
        self.count=0
        self.age_count={}
        self.age_sum={}

        # Load the model
        print('Loading feature extraction model')
        with open(self.SVCpath, 'rb') as infile:
            (self.classifymodel, self.class_names) = pickle.load(infile)
        print('Loaded classifier model from file "%s"' % self.SVCpath)

        # Get input and output tensors
        '''
        facenet.load_model(self.sess,self.facenet_model_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        '''
        load_recognition_model(self.recognition_graph,self.recognition_sess,self.mobilenet_model_path)
        self.images_placeholder=self.recognition_graph.get_tensor_by_name("input:0")
        self.embeddings=self.recognition_graph.get_tensor_by_name("embeddings:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        self.Database=np.load(self.database_path)
        self.detector=YOLO(self.yolo_model_path)
        self.age_gender_classifier=AgeGenderClassfier(self.age_gender_model_path)

        self.setupUi(self)                          # 加载窗体
        #以下是将按钮和功能联系起来
        self.btn_close.clicked.connect(self.close)   # 关闭程序
        self.btn_local_camera.clicked.connect(self.get_local_camera)   # 打开本地相机
        self.btn_from_video.clicked.connect(self.get_from_video)

        self.btn_get_faces.clicked.connect(self.get_faces)              # 得到人脸图像
        self.btn_delete_face.clicked.connect(self.delete_faces)                    # 报错
        self.btn_train_classifier.clicked.connect(self.train_classifier)                # 新建人脸数据

        self.timer_camera.timeout.connect(self.openFrame)  # 计时结CAM_NUM束调用open_frame方法
        self.timer_video.timeout.connect(self.openFrame)    #计时结束调用open_frame方法
        self.time_start=time.time()
        self.time_end=time.time()

    def openFrame(self):
        self.time_end=time.time()
        print ("allt time",self.time_end-self.time_start)
        t=self.time_end-self.time_start
        fps=cv2.getTickFrequency()/t
        self.time_start=time.time()
        t0=time.time()
        ret,frame = self.cap.read()
        if(self.cap.isOpened()):
            ts = cv2.getTickCount()
            ret, frame = self.cap.read()
            if ret:
                t1=time.time()
                height=frame.shape[0]
                width=frame.shape[1]
                image=np.array(frame)
                t2=time.time()
                boxes_c=self.detector.detect_image(frame)
                t3=time.time()
                imgs=[]
                ag_imgs=[]
                if boxes_c.shape[0]>0:
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]#检测出的人脸区域，左上x，左上y，右下x，右下y
                        score = boxes_c[i, 4]#检测出人脸区域的得分
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        margin=int((bbox[3]-bbox[1])*0.25)
                        x1=np.maximum(int(bbox[0]-margin),0)
                        y1=np.maximum(int(bbox[1]-margin),0)
                        x2=np.minimum( int(bbox[2]+margin),width)
                        y2=np.minimum( int(bbox[3]+margin),height)
                        crop_img=image[y1:y2,x1:x2]

                        ag_margin=int((bbox[3]-bbox[1])*0.4)
                        ag_x1=np.maximum(int(bbox[0]-ag_margin),0)
                        ag_y1=np.maximum(int(bbox[1]-ag_margin),0)
                        ag_x2=np.minimum( int(bbox[2]+ag_margin),width)
                        ag_y2=np.minimum( int(bbox[3]+ag_margin),height)
                        ag_crop_img=image[y1:y2,x1:x2]

                        #cv2.imwrite("tmp.png",crop_img)
                        scaled=misc.imresize(crop_img,(112,112),interp='bilinear')
                        img=load_image(scaled,False, False,112)
                        self.count+=1                       
                        imgs.append(img)
                        ag_img=misc.imresize(ag_crop_img,(160,160),interp='bilinear')
                        ag_img=load_image(ag_img,False, False,160)
                        ag_imgs.append(ag_img)

                        #当添加人脸时，把检测到的人脸保存在文件夹下
                        if self.add_face:
                            print ("添加人脸剩余时间",self.add_face_end-time.time())
                            output=misc.imresize(crop_img,(112,112))
                            cv2.imwrite(self.add_path+"/"+"%d.png" %(int(time.time()*100)),output)
                            #cv2.imwrite("train_output/"+"%d.png" %(int(time.time()*100)),output)
                            if time.time()>self.add_face_end:#停止添加人脸
                                self.timer_camera.stop()
                                self.label_camera.clear()
                                self.add_face=False
                                self.count=0
                                self.video_output.release()
                                self.btn_get_faces.setText("添加人脸数据")
                    #下面进行人脸识别，年龄检测，性别检测
                    if not self.add_face:
                        feed_dict = { self.images_placeholder:imgs}
                        embvecor=self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
                        embvecor=np.array(embvecor)

                        #利用SVM对人脸特征进行分类
                        predictions = self.classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lables=self.class_names[best_class_indices]
                        #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        face_num=boxes_c.shape[0]
                        for i in range(face_num):
                            best_class_probability=predictions[i][best_class_indices[i]]
                            print(best_class_probability)
                            if best_class_probability<0.4:
                                tmp_lables[i]="others"

                        ages,genders=self.age_gender_classifier.classify(ag_imgs)
                        for i in range(face_num):
                            if tmp_lables[i] not in self.age_count.keys():
                                self.age_sum[tmp_lables[i]]=ages[i]
                                self.age_count[tmp_lables[i]]=1
                            else:
                                self.age_sum[tmp_lables[i]]+=ages[i]
                                self.age_count[tmp_lables[i]]+=1                               
                            bbox = boxes_c[i, :4]#检测出的人脸区域，左上x，左上y，右下x，右下y
                            score = boxes_c[i, 4]#检测出人脸区域的得分
                            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                            gender="F" if genders[i] == 0 else "M"
                            age=int(self.age_sum[tmp_lables[i]]/self.age_count[tmp_lables[i]])
                            cv2.putText(frame, '{},{},{}'.format(tmp_lables[i],gender,int(age)), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)                                

                t4=time.time()
                cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255), 2)
                output=frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data,  width, height, bytesPerLine, 
                                 QtGui.QImage.Format_RGB888).scaled(self.label_camera.width(), self.label_camera.height())
                self.video_output.write(output)
                #cv2.imwrite("tmp.jpg",frame)
                #self.label_camera.setPixmap(QtGui.QPixmap.fromImage(q_image)) 
                t5=time.time()
                print ("before detect",t2-t1,"detect",t3-t2,"recognite",t4-t3,"show",t5-t4,"total",t5-t1)
            else:
                self.cap.release()
                self.video_output.release()
                self.timer_camera.stop()   # 停止计时器
                self.timer_video.stop()
                self.btn_from_video.setText(u'打开视频')

    # 获取本地摄像头视频
    def get_local_camera(self):
        if self.timer_video.isActive() or self.add_face:      # 查询网络摄像头是否打开,或者是否正在添加人脸
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请先关闭视频或者等待添加人脸完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        elif not self.timer_camera.isActive():
            self.cap=cv2.VideoCapture(self.CAM_NUM)

            flag = self.cap.isOpened()
            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.age_count={}
                self.age_sum={}
                size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print ("video fps",video_fps)
                self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), 12.5, size,1)
                self.timer_camera.start(0)     # 30ms刷新一次
                self.btn_local_camera.setText(u'关闭本地摄像头')

        else:
            self.timer_camera.stop()    # 定时器关闭
            self.cap.release()          # 摄像头释放
            self.label_camera.clear()   # 视频显示区域清屏RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
            self.video_output.release()
            #self.graphicsView.show()
            self.btn_local_camera.setText(u'打开本地摄像头')

    def get_from_video(self):
        if self.timer_camera.isActive() or self.add_face:#查询video是否打开
            QtWidgets.QMessageBox.warning(self, u"Warning", u"请先关闭摄像头或者等待添加人脸完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        elif not self.timer_video.isActive():                      
            videopath,ok= QtWidgets.QFileDialog.getOpenFileName(self,"getOpenFileName","video") 
            if ok:
                self.age_count={}
                self.age_sum={}
                print (videopath)
                self.cap=cv2.VideoCapture(videopath)

                size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), video_fps, size,1)
                if self.cap is None:
                    QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测视频格式是否正确", buttons=QtWidgets.QMessageBox.Ok,
                                    defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_video.start(0)
                    self.btn_from_video.setText(u"关闭视频")
        else:
            self.timer_video.stop()
            self.cap.release()
            #self.video_output.release()
            self.label_camera.clear()
            self.btn_from_video.setText(u'打开视频')



    def get_faces(self):
        name,add_ok=QInputDialog.getText(self,"请输入要添加的人脸姓名","",QLineEdit.Normal,"")
        if add_ok:
            self.add_path=os.path.join(self.add_picture_path,name)
            if os.path.exists(self.add_path):
                save_ok=QtWidgets.QMessageBox.warning(self, u"Warning", u"人脸库中已经包含该姓名，是否删除原人脸", buttons=QtWidgets.QMessageBox.Save| QMessageBox.Discard | QMessageBox.Cancel,
                                    defaultButton=QtWidgets.QMessageBox.Save)
                if save_ok==QMessageBox.Cancel:
                    print ("取消添加人脸")
                    return
                if save_ok==QMessageBox.Discard:
                    print ("放弃保存之前的人脸")
                    shutil.rmtree(self.add_path)
                    os.mkdir(self.add_path)
            else:
                os.mkdir(self.add_path)
            #flag = self.cap.open(self.CAM_NUM)
            #self.cap=cv2.VideoCapture("/home/zhengcy/Real-time-face-recognition-master/test/video_test.mp4")
            self.cap=cv2.VideoCapture(self.CAM_NUM)
            size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_output = cv2.VideoWriter(str(time.time()*1000)+".avi",  cv2.VideoWriter_fourcc(*'XVID'), video_fps, size,1)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.add_face=True
                self.btn_get_faces.setText("正在添加人脸")
                self.add_face_begin=time.time()+3
                while(time.time()<self.add_face_begin):
                    time.sleep(0.5)
                    #cv2.putText(frame, "人脸录制将在 " +"{:.3f}".format(time_begin-time.time())+"之后开始", (250, 160), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    #    (255, 0, 255), 2)
                    self.label_camera.setText("人脸录制将在 " +"{:.3f}".format(self.add_face_begin-time.time())+"之后开始")
                self.timer_camera.start(0)
                self.add_face_end=self.add_face_begin+15

    def delete_faces(self):
        name,add_ok=QInputDialog.getText(self,"请输入要删除的人脸姓名","请输入要删除的人脸姓名",QLineEdit.Normal,"")
        if add_ok:
            delete_path=os.path.join(self.add_picture_path,name)
            if not os.path.isdir(delete_path):
                QtWidgets.QMessageBox.warning(self, u"Warning", u"人脸库中不存在该姓名", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                shutil.rmtree(delete_path) 
                QtWidgets.QMessageBox.warning(self, u"Warning", u"成功删除人脸", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)     



    def debug(self):
        self.face_recog.reload_data()  # 重载人脸数据集
        num = self.face_recog.max_num
        file_names = self.face_recog.names
        print(file_names)
        if num > 0:
            result, ok = QtWidgets.QInputDialog.getItem(self, u"人脸数据校验", u"把人脸数据存入对应的文件夹中，可增加人脸识别的准确性。确定把图片存放在以下文件夹中吗？",
                                                        file_names, 1, False)
            if ok:
                if self.face_photo is not None:
                    # 保存图片
                    s_time = time.ctime().replace(' ', '_').replace(':', '_')
                    cv2.imwrite('./faces/' + result + '/' + str(s_time) + '.jpg', self.face_photo)
                    self.textEdit.append("已保存在./faces/" + result + '文件夹下!!')
        else:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"数据集为空,请新建人脸数据！", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    def train_classifier(self):
        augmentation(self.add_picture_path,self.train_picture_path)
        self.face2database()
        print("Train Classifier")
        #database_path为人脸数据库
        #SVCpath为分类器储存的位置
        Database=np.load(self.database_path)
        name_lables=Database['lab']
        embeddings=Database['emb']
        name_unique=np.unique(name_lables)
        labels=[]
        for i in range(len(name_lables)):
            for j in range(len(name_unique)):
                if name_lables[i]==name_unique[j]:
                    labels.append(j)
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(embeddings, labels)
        with open(self.SVCpath, 'wb') as outfile:
            pickle.dump((model,name_unique), outfile)
            print('Saved classifier model to file "%s"' % self.SVCpath)
        reply = QtWidgets.QMessageBox.information(self,
                                    "information",  
                                    "分类器训练完成。",  
                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def face2database(self,batch_size=90,image_size=112):
        #提取特征到数据库
        #picture_path为人脸文件夹的所在路径
        #model_path为facenet模型路径
        #database_path为人脸数据库路径
        dataset = facenet.get_dataset(self.train_picture_path)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False,image_size)
            feed_dict = { self.images_placeholder:images}
            emb_array[start_index:end_index,:] = self.recognition_sess.run(self.embeddings, feed_dict=feed_dict)
        np.savez(self.database_path,emb=emb_array,lab=labels)
        print("数据库特征提取完毕！")
        #emb_array里存放的是图片特征，labels为对应的标签


if __name__ == "__main__":
    if not os.path.exists("./faces"):
        os.makedirs("./faces")
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyDesignerShow()    # 创建实例
    myshow.show()           # 使用Qidget的show()方法
    sys.exit(app.exec_())

>>>>>>> origin/master
