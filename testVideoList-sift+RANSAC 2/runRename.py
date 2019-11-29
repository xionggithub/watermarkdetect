import os
import shutil

def wmdr_rename():
    src_dir = "new_video"
    src_full_dir = os.getcwd()+"/"+src_dir+"/"
    files = os.listdir(src_dir)
    count = len(files)
    adjustCount = 1
    while count > 10:
        adjustCount += 1
        count = count/10

    index = 0

    for file in files:
        index += 1
        file_dir = src_full_dir +file
        if os.path.isfile(file_dir) and (not file.startswith('.')):
            strings = file.split(".")
            print(strings)
            fileType = strings[len(strings) - 1]
            newFileName = src_full_dir+ str(index).rjust(adjustCount+1,"0")+"."+fileType
            os.rename(file_dir,newFileName)

    return

if __name__ == '__main__':
	wmdr_rename()
