
cd $ATCISELWORKDIR

datain="$ATCISELWORKDIR/resources/datasets"
out="$ATCISELWORKDIR/resources/outsel"

mkdir -p $out

datasets=(aisopos_ntua_2L)
methods=(cnn) # enn icf lssm lsbo ldis cdis xldis drop3 psdsp) 

for d in ${datasets[@]};
do
    echo $d ; 
    for method in ${methods[@]} 
    do
        echo $method ;
        python3 run\_generateSplit.py -d $d -m $method --datain $datain --out $out;
    done;
done;