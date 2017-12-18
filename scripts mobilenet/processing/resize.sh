#!/bin/sh

LOCAL_IMAGES="isic/images/"
LOCAL_TARGET="images224/"
SIZE="224"

for i in $LOCAL_IMAGES*.jpg; do
echo $i;
	convert $i -resize "($SIZE)x($SIZE)"\! $LOCAL_TARGET$(basename $i)
done