from robot_saeid import Robot
import numpy as np
import time
from simulation import vrep
import cv2
import matplotlib.pyplot as plt
import random
import time
import os
import shutil
from datetime import datetime



is_sim =True
workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
robot = Robot(is_sim, workspace_limits)

class StackingScene:
	
	def __init__(self,num_objects):
		robot.restart_sim()
		self.num_objects = num_objects
		self.object_list = ['Cuboid']+['Cuboid'+str(i) for i in range(num_objects-1)]
		self.run_trial_stability()		
		
		

	def get_object_position(self, object_name):
		sim_ret, handle  =vrep.simxGetObjectHandle(robot.sim_client,object_name,vrep.simx_opmode_blocking)
		_, cuboid_position = vrep.simxGetObjectPosition(robot.sim_client,handle,-1,vrep.simx_opmode_blocking)
		#print ('The location of '+object_name+' is:')
		return cuboid_position

	def set_object_position(self, object_name,x,y,z):
		sim_ret, handle  =vrep.simxGetObjectHandle(robot.sim_client,object_name,vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(robot.sim_client,handle,-1,(x,y,z),vrep.simx_opmode_blocking)

	def update_object_position(self,object_name):
		globals()['self.'+object_name+'_position'] = self.get_object_position(object_name)

	def take_image(self,image_name,sensor_name='Vision_sensor'):
		res, v1 = vrep.simxGetObjectHandle(robot.sim_client, sensor_name, vrep.simx_opmode_oneshot_wait)
		print ('Taking image')
		#print (v1)
		while (vrep.simxGetConnectionId(robot.sim_client) != -1):

			err, resolution, image = vrep.simxGetVisionSensorImage(robot.sim_client, v1, 0,  vrep.simx_opmode_blocking)

			#if err == vrep.simx_return_ok:
			#print (image)
			#print (err)
			#img = np.array(image,dtype=np.uint8)
			img = np.array(image,dtype=np.uint8)
			img.resize([resolution[1],resolution[0],3])
			print (img.shape)
			#print (img)
			#plt.imshow(img)
			#img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.imwrite(image_name,img)
			#if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		######## Adding a test of the image creation

	def set_the_scene(self):
		# Select the random location of the first item
		print (self.object_list)
		first_object = random.choice(self.object_list)
		x,y,z = self.get_object_position(first_object)
		self.set_object_position(first_object,0,0,0.01)
		self.object_list.remove(first_object)

		#print (temp)
		for i,obj in enumerate(self.object_list):
			self.set_object_position(obj,0,0,0.01*(i+1))
		print ('Done')

	def run_trial_stability(self):
		vrep.simxStopSimulation(robot.sim_client, vrep.simx_opmode_blocking)
		self.set_the_scene()
		positions_now = np.array([self.get_object_position(obj) for obj in self.object_list])
		
		vrep.simxStartSimulation(robot.sim_client, vrep.simx_opmode_blocking)
		self.take_image('new2.jpg') # has to be after running the simulation
		time.sleep(1)
		#time.sleep(5)
		positions_next = np.array([self.get_object_position(obj) for obj in self.object_list])
		vrep.simxStopSimulation(robot.sim_client, vrep.simx_opmode_blocking)
		now = datetime.now() 

		date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
		os.rename('new2.jpg',date_time+'.jpg')
		if np.linalg.norm((positions_now - positions_next), ord=1)>0.01:
			print ('unstable')
			shutil.move(date_time+'.jpg','./data/unstable')
		else:
			print ('stable')
			shutil.move(date_time+'.jpg','./data/stable')
		# else return stable


#

#take_image()
#robot.restart_sim()




def main():

	obj = StackingScene(4)




if __name__ == '__main__':
	main()