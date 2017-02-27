import tensorflow as tf
import numpy as np
import math
from data import input_data
from data import tag_data

input_num=100
data_len=2
output_num=1

test=[]
for i in range(100):
	test.append([1,100])


def structure(input_val,hidden1_unit,hidden2_unit):
	with tf.name_scope('hidden1'):
		w=tf.Variable(tf.truncated_normal([data_len,hidden1_unit],name='w'))
		b=tf.Variable(tf.zeros([hidden1_unit]),name='b')
		hidden1=tf.nn.relu(tf.matmul(input_val,w)+b)
	with tf.name_scope('hidden2'):
		w=tf.Variable(tf.truncated_normal([hidden1_unit,hidden2_unit],name='w'))
		b=tf.Variable(tf.zeros([hidden2_unit]),name='b')
		hidden2=tf.nn.relu(tf.matmul(hidden1,w)+b)
	with tf.name_scope('softmax'):
		w=tf.Variable(tf.truncated_normal([hidden2_unit,2],name='w'))
		b=tf.Variable(tf.zeros([2]),name='b')
		logits=tf.matmul(hidden2,w)+b
	return logits

def lossfunc(logits,tags):
	tags=tf.to_int64(tags)
	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,tags,name='xentropy')
	loss=tf.reduce_mean(cross_entropy,name='xentropy_mean')
	return loss

def training(loss,learning_rate):
	optimizer=tf.train.GradientDescentOptimizer(learning_rate)
	global_step=tf.Variable(0,name='global_step',trainable=False)
	train_op=optimizer.minimize(loss,global_step=global_step)
	return train_op

def placeholder_inputs():
	input_ph=tf.placeholder(tf.float32,shape=[input_num,data_len])
	tag_ph=tf.placeholder(tf.float32,shape=[input_num])
	return input_ph,tag_ph


def fill_feed_dict(input_feed,tag_feed,input_ph,tag_ph):
	feed_dict={input_ph:input_feed,tag_ph:tag_feed}
	return feed_dict

def run_training():
	graph=tf.Graph()
	with graph.as_default():	
		input_ph=tf.placeholder(tf.float32,shape=[input_num,data_len],name='input_ph')
		tag_ph=tf.placeholder(tf.float32,shape=[input_num],name='tag_ph')
		logits=structure(input_ph,100,100)
		loss=lossfunc(logits,tag_ph)
		train_op=training(loss,0.5)
		init=tf.initialize_all_variables()
		sess=tf.Session(graph=graph)
		sess.run(init)
		for step in range(100):
			feed_dict=fill_feed_dict(input_data,tag_data,input_ph,tag_ph)
			loss_value=sess.run([train_op,loss,logits],feed_dict=feed_dict)
			if step%2==0:
				print('step %d:loss=%.2f'%(step,loss_value[1]))
        logits=sess.run(logits,feed_dict={input_ph:test})
        print(logits)


if __name__=='__main__':
	run_training()
	






	
