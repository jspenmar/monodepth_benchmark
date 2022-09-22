#!/bin/bash

echo "-> Exporting Kitti Eigen-Zhou depth hints"
python api/data/export_kitti_hints.py --split eigen_zhou --mode train
python api/data/export_kitti_hints.py --split eigen_zhou --mode val

echo "-> Exporting Kitti Eigen depth hints"
python api/data/export_kitti_hints.py --split eigen --mode train
python api/data/export_kitti_hints.py --split eigen --mode val

echo "-> Exporting Kitti Eigen-Benchmark depth hints"
python api/data/export_kitti_hints.py --split eigen_benchmark --mode train
python api/data/export_kitti_hints.py --split eigen_benchmark --mode val
