#!/bin/bash


fileName=""
fileType=""
root_dir="./new_video"
rm  -r "no_mp4_video"
mkdir "no_mp4_video"
no_mp4_video_dir="../no_mp4_video"


function splitString(){
	str=$1
	OLD_IFS="$IFS" 
	IFS="." 
	arr=($str) 
	IFS="$OLD_IFS" 
	length=${#arr[@]}
	fileName=${arr[0]}
	fileType=${arr[$length-1]}
}


function transformVideo(){
    fileName=""
    fileType=""
	splitString $1
	if [ $fileType != "mp4" ]; then
		echo $fileType" is  not mp4 file"
		ffmpeg -i $fileName"."$fileType $fileName".mp4"
		cp $fileName"."$fileType  $no_mp4_video_dir
		rm $fileName"."$fileType
	fi
	
}
# 遍历文件 并处理
function runVideoListTest(){
	# for element in `ls $1`
	cd $1
	for element in `ls`
	do
		# dir_or_file=$1"/"$element
		dir_or_file=$element
        if [ -d $dir_or_file ]
        then 
            # getdir $dir_or_file
            echo $element 是目录：$dir_or_file
            # rm -r $dir_or_file
        else
           # echo $element 是文件：$dir_or_file
            #开始测试文件水印定位
            transformVideo $element
        fi

	done	
}

runVideoListTest $root_dir

