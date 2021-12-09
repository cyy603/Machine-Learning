import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

list1 = [i for i in range(9)]
print(list1)
list2 = [(i, j) for i in range(3) for j in range(4)]
print(list2)
list3 = [('第一列 *2：'+str(i*2), '第二列 +2：'+str(j+2)) for i, j in list2]
print(list3)
pp.pprint(list1)
pp.pprint(list2)
pp.pprint(list3)

print(list1[:5])
print(list1[:-1])

list4 = [2 * list1[i] if i % 2 == 1 else list1[i] for i in list1]
print(list4)

my_tuple = (10, 20, 30)
a, b, c = my_tuple
print(f"a={a}, b={b}, c={c}")
for obj in enumerate(my_tuple):
    print(obj)

print(np.sin(np.pi / 2))
print(np.sqrt(2))
print(np.power(3, 2) == pow(3, 2))

arr = np.random.random((1,10000))
print(len(arr))
print([np.mean(arr), np.std(arr)])

x1_values = np.arange(0, 2, 0.001)  
x2_values = x1_values[0:int(1/0.001)]
y1_values = x1_values * 2
y2_values = x1_values ** 2
y3_values = np.sin(2*np.pi*x2_values)

plt.figure()
plt.plot(x1_values, y1_values, label='$y=2x$')
plt.legend()

plt.figure()
plt.plot(x1_values, y2_values, label='$y=x^2$')
plt.legend()

plt.figure(num = 4, figsize=(8,5))
plt.plot(x2_values, y3_values, label='$y=sin(2 \pi x)$')
plt.legend()

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$y=f(x)$')
plt.show()

#def plot_sub_fig(x, y, index, ax):
