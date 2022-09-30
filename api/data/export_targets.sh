#!/bin/bash

echo "-> Exporting Kitti Eigen targets"
python api/data/export_kitti_targets.py --split eigen --mode test --use-velo-depth 1 --save-stem targets_test

echo "-> Exporting Kitti Eigen Zhou targets"
python api/data/export_kitti_targets.py --split eigen_zhou --mode test --use-velo-depth 0 --save-stem targets_test

echo "-> Exporting Kitti Eigen Benchmark targets"
python api/data/export_kitti_targets.py --split eigen_benchmark --mode test --use-velo-depth 0 --save-stem targets_test
