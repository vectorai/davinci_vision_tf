import os

categories = os.listdir('./')

outstring=''
for i,category in enumerate(categories):
        if category == 'make_train.py':
		continue
	middle_dir = os.listdir(category)
	temp_path=category+'/'+middle_dir[0]+'/'
	files = os.listdir(temp_path)
	for file_ in files:
		is_png = file_.find('.png')
		if is_png!=-1 and file_.find('mask')==-1 and file_.find('depth')==-1:
			infile=temp_path+file_
			maskfile = temp_path+file_[:is_png]+'_mask.png'
			lab = str(i+1)
			outstring += infile+' '+maskfile+' '+lab+'\n'
			print maskfile
with open('train.txt','wb') as f:			
	f.write(outstring)
