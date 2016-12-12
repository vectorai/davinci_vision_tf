import os
import scipy.io as sio
from PIL import Image
CATEGORY=0
INSTANCE=1
TOP=2
BOTTOM=3
LEFT=4
RIGHT=5

matbb = sio.loadmat('table_small_2.mat')['bboxes']

for zind in range(matbb.shape[1]):
    data=matbb[0,zind]
    # filename = data[zind]["filename"]
    # print filename
    print(zind)
    tail = 'table_small_2/table_small_2_' + str(zind + 1) + '.png'
    head, tail = os.path.split(tail)
    basename, file_extension = os.path.splitext(tail)    
    f = open(os.path.join(head,basename) + '.xml','w') 
    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t\t<filename>' + tail + '</filename>' + '\n'
    f.write(line)
    line = '\t\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
    f.write(line)
    im=Image.open(os.path.join(head , tail))
    (width, height) = im.size
    line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + str(height) + '</height>\n\t'
    line += '\t<depth>Unspecified</depth>\n\t</size>'
    f.write(line)
    line = '\n\t<segmented>Unspecified</segmented>'
    f.write(line)
    ind = 0
    while ind < len(data):
        line = '\n\t<object>'
        line += '\n\t\t<name>'+str(data[ind][0][CATEGORY][0])+'</name>\n\t\t<pose>Unspecified</pose>'
        line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>Unspecified</difficult>'
        xmin = (data[ind][0][LEFT][0][0])
        line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
        ymin = (data[ind][0][TOP][0][0])
        line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
        xmax = (data[ind][0][RIGHT][0])
        ymax = (data[ind][0][BOTTOM][0])
        xmax = xmin + width
        ymax = ymin + height
        line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
        line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
        line += '\n\t\t</bndbox>'
        line += '\n\t</object>'     
        f.write(line)
        ind +=1
    line = '\n</annotation>'
    f.write(line)
    f.close()
    zind +=1
