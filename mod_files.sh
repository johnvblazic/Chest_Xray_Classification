python generate_mod_files.py > files_to_mod.txt

cat files_to_mod.txt | while read line
do
	mogrify -colorspace Gray $line
done