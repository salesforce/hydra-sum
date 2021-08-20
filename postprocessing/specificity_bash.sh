INPUT_DIR=../../data/xsum/model-bart-3heads-8layers/3/head1
filename=$(date +%s)
python specificity.py --input_dir $INPUT_DIR  --output_file $filename  --function write_files

cd ../speciteller

outfile=$(echo "$filename"out)
echo $outfile

## use some tool to get sentence level specificity
#python speciteller.py --inputfile ../postprocessing/$filename --outputfile ../postprocessing/$outfile

filename_gen=$(echo "$filename"gen)
outfile_gen=$(echo "$filename_gen"out)
python speciteller.py --inputfile ../postprocessing/$filename_gen --outputfile ../postprocessing/$outfile_gen

python ../postprocessing/specificity.py --input_dir $INPUT_DIR  --input_file ../postprocessing/$filename  --function combine_scores
rm ../postprocessing/$filename ../postprocessing/$outfile
rm ../postprocessing/$filename_gen ../postprocessing/$outfile_gen