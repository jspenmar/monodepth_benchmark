#!/bin/bash
# Script to download the SYNS-Patches images.

key=6407c34a-39af-448c-8ab9-c74c9f1eef35

wget -c 'https://codalab.lisn.upsaclay.fr/my/datasets/download/'$key
unzip -o $key
rm $key
