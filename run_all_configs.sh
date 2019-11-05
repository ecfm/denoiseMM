#!/bin/bash
CONF_DIR="../configs"
LOG_DIR="../logs"
LAST=''
#LAST=(conf_b0_cuda0_144 conf_b0_cuda1_193 conf_b0_cuda2_178 conf_b0_cuda3_167)
START_GPU=$1
if [[ -z "$2" ]]; then
    GPUS=$(( START_GPU + 1 ))
else
    GPUS=$2
fi
run_with_cuda() {
    START='false'
    for filename in `ls -d ${CONF_DIR}/"conf_cuda$1"* | sort -V`; do
        FILE_BASE=`basename ${filename%.*}`
        if [[ ${LAST} != '' && ${START} == 'false' ]]; then
            if [[ ${LAST[$1]} == ${FILE_BASE} ]]; then
                 START='true'
            fi
            continue
        fi
        COUNT=`ps -o command -C python  | grep "main.py" | grep "cuda$1" | sort | uniq | wc -l`
        while [[ ${COUNT} -ge 1 ]]; do
             sleep 5
             COUNT=`ps -o command -C python  | grep "main.py" | grep "cuda$1" | sort | uniq | wc -l`
        done
        LOG_FILE="$LOG_DIR/${FILE_BASE}.log"
        git log -1 > ${LOG_FILE}
        git diff -- '*.py' >> ${LOG_FILE}
        echo "running with $filename"
        python -u main.py "$filename" >> ${LOG_FILE}  #&
    #    sleep 30
    done
}

for ((i= $((START_GPU)); i<$(( GPUS ));i++)); do
    run_with_cuda ${i} 2>&1 > ../run_all_${i}.log &
done