for k in 4 3 0
do
    for j in 0 1
    do
        python newtk.py $j $k | tee "new.pe.evaluation.${j}.${k}"
    done
done