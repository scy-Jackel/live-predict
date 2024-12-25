import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import spline

# def avg(lin,wid):
#     lout = []
#     lin = np.array(lin)
#     sum = np.sum(lin[:wid])
#     for k in range(wid,len(lin)):
#         lout.append(float(sum)/float(wid))
#         sum -= lin[k - wid]
#         sum += lin[k]
#     return lout

def avg(lin,wid):
    lout = []
    lin = np.array(lin)
    sum = np.sum(lin[:wid])
    for k in range(wid,len(lin)):
        lout.append(float(sum)/float(wid))
        sum -= lin[k - wid]
        sum += lin[k]
    return lout

csv_path0 = './loss_record.csv'
csv_reader = csv.reader(open(csv_path0))
l0 = []
l1 = []

for item in csv_reader:
    if len(item) == 6:
        step = int(item[0])
        live = item[3]
        if step>len(l0):
            l0.append(float(item[1]))
            l1.append(live)
        else:
            l0[step - 1] = float(item[1])
            l1[step - 1] = live

live_error = []
dead_error = []
for k in range(len(l0)):
    if int(l1[k])==0:
        dead_error.append(l0[k])
    else:
        live_error.append(l0[k])

# print len(dead_error)
# print len(live_error)

dead_error = avg(dead_error,1000)
live_error = avg(live_error,1000)
num = 44000
epo = np.linspace(0, 350, num)
dead_error = dead_error[:num]
live_error = live_error[:num]


plt.ylim(0., 5.)
plt.xlim(-1, 350)
plt.plot(epo,dead_error , color='gray',label = 'Dead')
plt.plot(epo,live_error , color='orange',label = 'Survivor')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.grid(True)
plt.savefig("./loss.png")
plt.show()