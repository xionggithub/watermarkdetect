#!/bin/bash

result_dir="result"
gray_result_dir="result/grayResult"

test_frames_dir="test_frames"

# 判断旧的定位结果文件夹是否存在，存在则删除
if [ ! -f $result_dir ];then
    echo ""
else
    rm -r $result_dir
fi

#创建新的定位结果文件夹
mkdir $result_dir
#创建新的灰度图定位结果文件夹
mkdir $gray_result_dir

#判断是否截取到视频帧
count=`ls $test_frames_dir | wc -w`
if [ $count == "0" ]; then
	#未截取到
	echo "未提取到视频帧!"
else

	#定位水印 使用sift + RANSAC 定位 准确度高
	python3 water_mask_find_sift+filterRANSAC.py
	#判断是否定位到，没定位到 移除测试文件夹
	if [ ! -f "./result/标记水印位置图.jpg" ];then
		if [ ! -f "./result/grayResult/标记水印位置图.jpg" ];then
			echo "sift+RANSAC                               未定位到水印!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # rm  -r $result_dir
		else
			echo "sift+RANSAC 定位水印成功!"
		fi
	else
		echo "sift+RANSAC 定位水印成功!"
	fi
fi
