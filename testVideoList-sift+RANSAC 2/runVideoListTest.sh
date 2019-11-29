#!/bin/bash
#指定路径
root_dir="./new_video"
runShAndPy="runPyAndShell"
find_mark_videos_reliability_high_dir="find_mark_videos_reliability_high/"
find_mark_videos_reliability_normal_dir="find_mark_videos_reliability_normal/"
testLeftLogoFileRoot_dir="mark_logos/mask_logo_left.png"
testRightLogoFileRoot_dir="mark_logos/mask_logo_right.png"
testLogoFileRoot_dir="/mask_logo.png"
#执行脚本
runSIFTANDRANSACPyShell="runSIFTAndRANSACPython.sh"
runSIFTInvertFilterPyShell="runSIFTAndInvertFilterPython.sh"

fileName=""
fileType=""
videoTestName="test."
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

function copyFiles(){
	srcDir=$1
	dstDir=$2

	for element in `ls $srcDir`
	do
		dir_or_file=$srcDir"/"$element
        if [ -f $dir_or_file ]
        then
        	cp $dir_or_file $dstDir
        fi
	done	
}

function removeFileOnly(){
	srcDir=$1
	for element in `ls $srcDir`
	do
		# dir_or_file=$1"/"$element
		dir_or_file=$srcDir"/"$element
        if [ -f $dir_or_file ]
        then
        	rm $dir_or_file
        fi
	done
}

function createDirAndMoveTestVideo(){
	#提取视频名称
	splitString $1

	# 创建目录并拷贝到目录中
    if [ ! -f $fileName ];then
        echo ""
    else
        rm -r $fileName
    fi

    #创建和视频同名称的临时文件夹
	mkdir $fileName
	#拷贝视频到该目录
	cp $1 $fileName/$videoTestName$fileType
	#拷贝模板
	cp "../"$testLeftLogoFileRoot_dir $fileName$testLogoFileRoot_dir

	# 拷贝执行脚本到该目录
	srcDir="../"$runShAndPy
	copyFiles $srcDir $fileName
	
	cd $fileName
	
	#执行正确率高的脚本
	./$runSIFTANDRANSACPyShell
	result_dir="result"
    if [ ! -d $result_dir ];then
    	
    	./$runSIFTInvertFilterPyShell
    	if [ ! -d $result_dir ];then
    		cd ../
    		#如果没找到水印，返回到上级目录后删除当前的临时目录
        	rm  -r  $fileName
    	else
    		cd ../
    		removeFileOnly $fileName
			find_mark_videos_dir="../"$find_mark_videos_reliability_normal_dir
			#如果找到水印，返回到上级目录后将该目录和视频拷贝到结果目录
			cp -r $fileName $find_mark_videos_dir$fileName
			cp    $fileName"."$fileType  $find_mark_videos_dir$fileName"."$fileType
			#拷贝完成就删除
	        rm  -r  $fileName
	        rm  $1
    	fi
    	
	else
		cd ../
        removeFileOnly $fileName
		find_mark_videos_dir="../"$find_mark_videos_reliability_high_dir
		#如果找到水印，返回到上级目录后将该目录和视频拷贝到结果目录
		cp -r $fileName $find_mark_videos_dir$fileName
		cp    $fileName"."$fileType  $find_mark_videos_dir$fileName"."$fileType
		#拷贝完成就删除
        rm  -r  $fileName
        rm  $1
	fi
    echo $fileName
    
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
            # echo $element 是目录：$dir_or_file
            rm -r $dir_or_file
        else
           # echo $element 是文件：$dir_or_file
            #开始测试文件水印定位
            createDirAndMoveTestVideo $element
           # rm $element
        fi

	done	
}

rm -r $find_mark_videos_reliability_high_dir
mkdir $find_mark_videos_reliability_high_dir
rm -r $find_mark_videos_reliability_normal_dir
mkdir $find_mark_videos_reliability_normal_dir

##先移除不完整文件
#./runcleanErrorVideoFile.sh
##再将非mp4 视频转MP4
#./runTranformVideo.sh
##将文件重新命名 便于统计测试数据
#python3 runRename.py
#开始定位
runVideoListTest $root_dir


