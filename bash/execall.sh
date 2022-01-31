
cd $ATCISELWORKDIR

nfolds=10
max_iter=20

methods=(bert xlnet roberta distilbert albert bart gpt2)
datasets=(aisopos_ntua_2L)

for deepmethod in ${methods[@]};
do

    for d in ${datasets[@]};
    do 

        base_or_large=base
        batch_size=16
        max_len=256

        if [[ "$deepmethod" == "bert" ]] || [[ "$deepmethod" == "distilbert" ]]; then 
            batch_size=32
        fi

        for f in `seq 0 9`;
        #for f in 0;
        do
            python run_deepBasedClassifiers.py --dataset $d \
            --nfolds $nfolds \
            --foldesp $f \
            --path_datasets resources/datasets/ \
            --out resources/outclf/deep/${nfolds}Fold/${deepmethod}/${base_or_large} \
            --pretrained_models_path $ATCISELWORKDIR/resources/pretrained_models/ \
            --deepmethod $deepmethod \
            --base_or_large ${base_or_large} \
            --batch_size ${batch_size} \
            --max_len ${max_len} \
            --max_iter ${max_iter} \
            --save_proba 1;
        done;

    done;
done; 
exit
