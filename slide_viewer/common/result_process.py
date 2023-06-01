import  numpy as np
import random
# t = np.random.rand(5,1,2)
# print(t)
# a = [[np.random.randint(i, j) for j in range(6, 9)] for i in range(0, 2)]
# # print(a)
map = []

for i in range(0, 3):

  for j in range(0, 4):
    map.append([i,j,np.random.randint(1, 9),np.random.uniform(0,1)])
a = map[1]
print(a[1]) ##去掉头尾
print(map)