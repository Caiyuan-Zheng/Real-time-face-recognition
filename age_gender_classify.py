import os
import cv2
import numpy as np
import time
import inception_resnet_v1
import tensorflow as tf


class AgeGenderClassfier():
    def __init__(self,model_path):
        self.graph=tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with self.sess.as_default():
                self.images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
                images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.images_pl)
                self.train_mode = tf.placeholder(tf.bool)
                age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                             phase_train=self.train_mode,
                                                                             weight_decay=1e-5)
                self.gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
                age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
                self.age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
                init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
                self.sess.run(init_op)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print("restore model!")
                else:
                    print("can't load pretrained model!")


    def classify(self,images):
        ages,genders = self.sess.run([self.age, self.gender], feed_dict={self.images_pl: images, self.train_mode: False})
        return ages,genders 