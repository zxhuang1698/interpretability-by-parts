#!/bin/bash

# need to install p7zip first
# sudo apt-get install p7zip-full

unzip -q ./img_align_celeba.zip
mv ./img_align_celeba ./aligned
7za x ./img_celeba.7z.001
mv ./img_celeba ./unaligned
unzip -q ./annotation.zip
rm ./img_align_celeba.zip
rm ./img_celeba.7z.*
