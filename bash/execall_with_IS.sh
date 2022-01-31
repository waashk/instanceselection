
cd $ATCISELWORKDIR

nfolds=10
max_iter=20
#dataset,deepmethod
for i in aisopos_ntua_2L,bert;
do 
    IFS=',' read d deepmethod <<< "${i}"
    echo "${d}" and "${deepmethod}"

        base_or_large=base
        batch_size=16
        max_len=256

        if [[ "$deepmethod" == "bert" ]] || [[ "$deepmethod" == "distilbert" ]]; then 
            batch_size=32
        fi

    for ismethod in cnn enn icf drop3 lssm lsbo ldis cdis xldis psdsp;
    #for ismethod in cnn;
	do

        #for f in `seq 0 9`;
        for f in 0;
        do
            python run_deepBasedClassifiers.py --dataset $d \
            --nfolds $nfolds \
            --foldesp $f \
            --path_datasets resources/datasets/ \
            --path_selection resources/outsel/selection/ \
			--out resources/outclf/deep/${nfolds}Fold/${deepmethod}/${base_or_large} \
            --pretrained_models_path $ATCISELWORKDIR/resources/pretrained_models/ \
            --deepmethod $deepmethod \
            --base_or_large ${base_or_large} \
            --batch_size ${batch_size} \
            --max_len ${max_len} \
            --max_iter ${max_iter} \
            --ismethod $ismethod ; #--save_model 1; --save_proba 1 #
        done;
    done;

done;

