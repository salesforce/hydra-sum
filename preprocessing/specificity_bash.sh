INPUT_FILE=../data/cnndm/cnn/test.tsv
OUTPUT_FILE=../data/cnndm/cnn/specificity/test.tsv
INTERMEDIATE_FILE=$(date +%s)
export PYTHONPATH=./
python preprocessing/specificity.py --input_file $INPUT_FILE  --intermediate_file $INTERMEDIATE_FILE  --function write_files

## use some specificity tool to get summary level specificity
#cd speciteller
#outfile=$(echo "$INTERMEDIATE_FILE"out)
#echo $outfile
#python speciteller.py --inputfile ../$INTERMEDIATE_FILE --outputfile ../$outfile

cd ../
export PYTHONPATH=./
python preprocessing/specificity.py --input_file $INPUT_FILE  --output_file $OUTPUT_FILE  --intermediate_file $INTERMEDIATE_FILE  --function combine_scores
rm $INTERMEDIATE_FILE $outfile $(echo "$INTERMEDIATE_FILE"_num)
