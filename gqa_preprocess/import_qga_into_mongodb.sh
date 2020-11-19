filename_list=`ls ./*.json`


# for fullfile in `ls ./*.json`; do
for fullfile in `ls ./train_balanced_questions.json`; do
    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    basename="${filename%.*}"
    outfile_tmp='tmp_'$basename'.json'
    outfile=$basename.ndjson
    cat $filename | jq -nc --stream '
    def atomize(s):
        fromstream(foreach s as $in ( {previous:null, emit: null};
        if ($in | length == 2) and ($in|.[0][0]) != .previous and .previous != null
        then {emit: [[.previous]], previous: $in|.[0][0]}
        else { previous: ($in|.[0][0]), emit: null}
        end;
        (.emit // empty), $in) ) ;
    atomize(inputs)' > $outfile_tmp
    jq -c '{"key_id": keys[]} + .[]' $outfile_tmp > $outfile
    mongoimport --drop --db gqa --collection $basename --file $outfile
    rm ./$outfile_tmp ./$outfile
done
