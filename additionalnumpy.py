#playing around with numpy packages
import numpy as np 
arr = np.array([[1,2,3],[2,4,6]
				[6,7,8],[9,10,1]])
#now getting the various paramters of the array
print(arr.size)
print(arr.ndim)
print(arr.size)#indicates the size of the array
print(arr.dtype)#it prints the type of the elements in the array, in this case it is Int
print(type(arr)) #prints the type of the array, in this case it is int32

#additional functions
arr = np.array([1,2,3],dtype='float') #transforms the numbers in the 
print(arr)
a = np.array((1,2,3)) #array from tuples
b = np.zeros((3,4))#array of zeros
d = np.full((3,3),6,dtype='complex')#array of 6, in complex form
e = np.random.random((2,2))
f = np.arange(0,x,y)#range of 0 to x , with steps of y
g = np.linspace(0,5,10)#range 0 to 5 with 10 values
h = arr.resphape(x,y,z)#reshaping arr in the shape of x.y.z
