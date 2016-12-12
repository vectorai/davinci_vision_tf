import os

outstring=''
scenes=os.listdir('./')
for z,scene in enumerate(scenes):
	if not os.path.isdir(scene):
		continue
	categories = os.listdir(scene)

	for i,category in enumerate(categories):
                temp_path= os.path.join(scene,category)
		if not os.path.isdir(temp_path):
			continue
		files = os.listdir(temp_path)
		for file_ in files:
			is_png = file_.find('.png')
			if is_png!=-1 and file_.find('depth')==-1 and os.path.exists(os.path.join(temp_path,file_[:is_png]+'.xml')):
				infile=os.path.join(temp_path,file_[:is_png])
                                outstring+=infile+'\n'
				print infile
with open('trainval.txt','wb') as f:			
	f.write(outstring)
