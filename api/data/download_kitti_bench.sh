#!/bin/bash
# Script to download the corrected Kitti depth maps used in the official Benchmark.

download_file () {
  local in_file=$1
  echo $in_file
  fullname=${in_file}.zip
  wget -c 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'$fullname
  unzip -o $fullname
  rm $fullname
}


for i in devkit_depth data_depth_selection data_depth_annotated; do
  download_file $i &
done
