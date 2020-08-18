#!/bin/bash
printf "Downloading zipped data from the web \n"

wget -O skin-cancer-ham10000.zip 'https://s3.amazonaws.com/aws-hcls-datasets/skin-cancer-ham10000.zip?X-Amz-SignedHeaders=host&X-Amz-Credential=AKIATKEJVGWCERFPJ5EW%2F20200818%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Expires=604800&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200818T150529Z&X-Amz-Signature=5c71a7d68ee606a8dc182ee7a69847a64cde72110b4206616361d13f843369a9'

printf "\n Unzipping data \n"
unzip skin-cancer-ham10000.zip -d ./