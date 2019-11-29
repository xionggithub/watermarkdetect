#!/bin/bash
# 遍历文件 并处理
function runVideoListTest(){
	# for element in `ls $1`
	cd $1
	errorVideos_dir= $2

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
#            echo $element 是文件：$dir_or_file
			fileSize=`ls -l $element | awk '{ print $5 }'`
			
			# echo "fileSize:"$fileSize
			
			minsize=$((348))
			
			if [ $fileSize -le $minsize ]
			then
			    echo "文件不完整"

			    cp  $element  $errorVideos_dir
			    rm  $element
			else 
				echo ""
			fi
        fi

	done	
}



root_dir="./new_video"
errorVideos_dir="./errorVideos"
rm -r $errorVideos_dir
mkdir $errorVideos_dir
errorVideos_dir="../errorVideos"
runVideoListTest $root_dir $errorVideos_dir
