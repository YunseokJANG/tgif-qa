#!/usr/bin/env bash
if [ "$1" == '' ] || [ "$2" == '' ]; then
    echo "Usage: $0 <input folder> <output folder>";
    exit;
fi
ext=gif
for file in "${1}"/*."${ext}"; do
    destination="${2}${file:${#1}:${#file}-${#1}-${ext}-1}";
    echo $destination
    mkdir -p "$destination";
    ffmpeg -i "$file" "$destination/%d.jpg";
done
