import numpy as np
import matplotlib.pyplot as plt

#1
print(np.__version__)
print(np.show_config())
#2
print(np.info(np.add))
#3
print(np.all([1,2,3,4]))
#4
print(np.any([0,0,0,1]))
#5
print(np.isfinite([0,1,np.nan,np.inf]))
#6
print(np.isinf([0,1,np.nan,np.inf]))
#7
print(np.isnan([0,1,np.nan,np.inf]))
#8
x = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
print(np.iscomplex(x))
print(np.isreal(x))
print(np.isscalar(3.1))
print(np.isscalar([3.1, 3.2]))
#9
#10
x = [3,5]
y = [2,5]
print(np.greater(x,y))
print(np.greater_equal(x,y))
print(np.less(x,y))
print(np.less_equal(x,y))
#11
#12
x = np.array([1,7,13,105], dtype=np.float64)
print(x.size, x.itemsize)
#13
print(np.ones(10), np.zeros(10), np.ones(10)*5)
#14
print(np.arange(30, 70))
#15
print(np.arange(30, 70, 2))
#16
print(np.identity(3))
#17
print(np.random.rand(1))
#18
print(np.random.normal(0, 1, 15))
#19
print(np.arange(15,56)[1:-1])
#20
for i in np.nditer(np.arange(0,12).reshape(3,4)):
    print(i)
#21
print(np.linspace(5, 50, 10))
#22
a = np.arange(21)
a[(a>=9)&(a<=15)] *= -1
print(a)
#23
print(np.random.randint(0, 11, 5))
#24
print(np.array([2,3])*np.array([4,4]))
#25
print(np.arange(10,22).reshape(3,4))
#26
print(np.arange(0,12).reshape(3,4).shape)
#27
print(np.identity(3))
#28
a = np.ones((10,10))
a[1:-1, 1:-1] = 0
print(a)
#29
print(np.diag([1,2,3,4,5]))
#30
a = np.zeros((4,4))
a[::2, 1::2] = 1
a[1::2, ::2] = 1
print(a)
#31
print(np.random.random((3,3,3)))
print(np.random.rand(3,3,3))
#32
a = np.arange(1,101).reshape(10,10)
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
#33
a1 = np.arange(0,6).reshape(2,3)
a2 = np.arange(0,6).reshape(3,2)
print(np.dot(a1, a2))
#34
a1 = np.arange(0,12).reshape(4,3)
a2 = np.array([1,2,3])
print(a1)
print(a2)
for r in range(len(a1)):
    a1[r,:] += a2
print(a1)
#35
a1 = np.arange(0,12).reshape(4,3)
print(a1)
np.save("35.npy", a1)
a2 = np.load("35.npy")
print(a2)
print(np.array_equal(a1, a2))
#36
#37
#38
#39
l = [1,2,3,4,5,6]
a = np.array(l)
print(type(l), type(a), type(a.tolist()))
#40
x = np.arange(0.0, 2*np.pi, 0.01)
y = np.sin(x)
# plt.plot(x,y)
# plt.show()
#41
df = np.float32(32.0)
f = df.item()
print(type(df), type(f))
#42
#43
a1 = np.array([[1,2,np.nan], [np.nan,3,4]])
print(np.isnan(a1))
#44
print(np.array_equal([1,2,3], [1,2,3]))
#45
#46
#47
print(np.random.rand(40))
#48
print(np.random.normal(200,7,40).reshape(8,5))
print(np.random.randn(8,5) * 7 + 200)
#49
print(np.random.choice(7, 5, p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
print(np.random.choice(7, 5, replace=False, p=[0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0]))
#50
a1 = np.random.randint(1,100,16).reshape(4,4)
a2 = a1[::-1]
print(a1)
print(a2)
#51
print(np.zeros((5,6)))
#52
a1 = np.random.randint(1,100,16).reshape(4,4)
print(a1)
print(np.sort(a1, axis=0))
print(np.sort(a1, axis=1))
#53
a1 = np.arange(0,16).reshape(4,4)
print([a1>5])
#54
a1 = np.arange(0,16).reshape(4,4)
print(np.where(a1 == 3, 1024, a1))
#55
a1 = np.arange(0,16).reshape(4,4)
print(np.zeros_like(a1))
#56
#57
#58
a1 = np.random.randint(1,100,16).reshape(4,4)
a2 = a1[::-1, ::-1]
print(a1)
print(a2)
#59
print(np.array([[0,1],[2,3]]) * np.array([[3,2],[1,0]]))

#1
print(np.__version__)
#2
print(np.array([1,2,3,4]))
#3
print(np.arange(2,11).reshape(3,3))
#4
a = np.zeros(10)
a[6] = 11
print(a)
#5
print(np.arange(12,38))
#6
print(np.arange(12,38)[::-1])
#7
print(np.array([1,2,3,4], dtype=np.float32))
print(np.asfarray([1,2,3,4]))
#8
a = np.ones([5,5])
a[1:-1, 1:-1] = 0
print(a)
#9
a = np.ones([5,5])
a[0, :] = 0
a[-1, :] = 0
a[:, 0] = 0
a[:, -1] = 0
print(a)
a = np.pad(a, pad_width=1, mode="constant", constant_values=0)
print(a)
#10
a = np.zeros((8,8))
a[0::2, 1::2] = 1
a[1::2, 0::2] = 1
print(a)
#11
print(np.array(([8,4,6], [1,2,3])))
#12
print(np.append(np.array([10,20,30]), [40,50,60,70,80,90]))
#13
a = np.empty((3,4))
print(a)
a = np.full((3,4), 6)
print(a)
#14
#15
#16
a = np.array([1,2,3,4]).reshape(2,2)
print(a.size, a.itemsize, a.nbytes)
#17
a1 = np.array([0,10,20,40,60])
a2 = np.array([0,40])
print(np.isin(a1, a2))
print(np.in1d(a1, a2))
#18
print(np.intersect1d(a1, a2))
#19
print(np.unique([10,10,20,20,30,30]))
print(np.unique([[1,1],[2,3]]))
#20
print(np.setdiff1d(a1, a2))
#21
#22
a1 = np.array([0,10,20,40,60,80])
a2 = np.array([10,30,40,50,70])
print(np.sort(np.append(a1, a2)))
print(np.union1d(a1,a2))
#23
print(np.all([1,2,3,4]))
print(np.all([0,2,3,4]))
#24
print(np.all([[1,2,3],[0,1,2]], axis=0))
#25
print(np.tile([1,0], 4))
#26
print(np.repeat([1,2,3], 2))
#27
print(np.max([[1,9,3], [4,5,0]], axis=0))
print(np.min([1,2,3,4,5,6], axis=0))
#28
a1 = np.array([1,2])
a2 = np.array([4,5])
print(np.greater(a1, a2))
print(np.greater_equal(a1, a2))
print(np.less(a1, a2))
print(np.less_equal(a1, a2))
#29
a = np.array([[4,6,0],[2,1,8]])
print(a)
print(np.sort(a, axis=0))
print(np.sort(a, axis=1))
#30
f_names = ("Betsey", "Shelley", "Lanell", "Genesis", "Margery")
l_names = ("Battle", "Brien", "Plotner", "Stahl", "Woolum")
print(np.lexsort((f_names, l_names)))
#31
