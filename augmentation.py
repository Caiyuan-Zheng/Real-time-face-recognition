import Augmentor
import os
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

