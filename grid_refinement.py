import numpy as np
import matplotlib.pyplot as plt

def getErrorVal(fileloc):
    f = open(fileloc + 'log.txt','r')
    linelist = f.readlines()
    IE = linelist[1]
    ME = linelist[2]
    area = linelist[4]
    IE = IE.split(' ')[3]
    ME = ME.split(' ')[3]
    area = area.split(' ')[3]
    return IE, ME, area

def computeOrder_3(err1, err2, err3):
    return (1/np.log(2))*np.log(np.abs((err3 - err2)/(err2-err1)))

def computeOrder_2(err1, err2):
    return (1/np.log(2))*np.log(np.abs((err2)/(err1)))

def main():
    gridlist = ['64\\', '128\\', '256\\', '512\\', '1024\\']
    IE = []
    ME = []
    area_list = []
    ie_3 = []
    me_3 = []
    area_3 = []
    ie_2 = []
    me_2 = []
    area_2 = []
    for i in range(len(gridlist)):
        IE.append(getErrorVal('n = ' + gridlist[i])[0])
        ME.append(getErrorVal('n = ' + gridlist[i])[1])
        area_list.append(getErrorVal('n = ' + gridlist[i])[2])
    f = open('errorLog.txt', 'w')
    f.write('Order 3 grids:\n')
    plt.figure()
    plt.title('3 grids')
    for j in range(1, 4):
        ie_3.append( computeOrder_3(float(IE[j+1]), float(IE[j]), float(IE[j-1])))
        me_3.append(computeOrder_3(float(ME[j+1]), float(ME[j]), float(ME[j-1])))
        area_3.append(computeOrder_3(float(area_list[j+1]), float(area_list[j]), float(area_list[j-1])))
        f.write('IE = ' + str(ie_3[j-1]) + '\n')
        f.write('ME = ' + str(me_3[j-1]) + '\n')
        f.write('area = ' + str(area_3[j-1]) + '\n\n')
    plt.plot(ie_3, 'r', label='Interface error')
    plt.plot(me_3, 'b', label='Average mass error')
    plt.plot(area_3, 'g', label='Area change')
    plt.legend()
    plt.savefig('errorplot3grids.png')
    plt.show()
    f.write('Order 2 grids:\n')
    for j in range(1, 5):
        ie_2.append( computeOrder_2(float(IE[j]), float(IE[j-1])))
        me_2.append(computeOrder_2(float(ME[j]), float(ME[j-1])))
        area_2.append(computeOrder_2(float(area_list[j]), float(area_list[j-1])))
        f.write('IE = ' + str(ie_2[j-1]) + '\n')
        f.write('ME = ' + str(me_2[j-1]) + '\n')
        f.write('area = ' + str(area_2[j-1]) + '\n\n')
    plt.plot(ie_2, 'r', label='Interface error')
    plt.plot(me_2, 'b', label='Average mass error')
    plt.plot(area_2, 'g', label='Area change')
    plt.legend()
    plt.savefig('errorplot2grids.png')
    plt.show()
        # f.write('IE = ' + str(computeOrder_2(float(IE[j]), float(IE[j-1]))) + '\n')
        # f.write('ME = ' + str(computeOrder_2(float(ME[j]), float(ME[j-1]))) + '\n')
        # f.write('area = ' + str(computeOrder_2(float(area_list[j]), float(area_list[j-1]))) + '\n\n')
    f.close()

if __name__=='__main__':
    main()