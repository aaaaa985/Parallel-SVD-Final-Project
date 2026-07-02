#!/bin/sh
#PBS -N qsub_mpi_hybrid
#PBS -e test_hybrid_np2_omp4.e
#PBS -o test_hybrid_np2_omp4.out
#PBS -l nodes=2:ppn=4

set -x

RUNDIR=/home/${USER}
MASTER=master_ubss1
MASTER_DIR=/home/${USER}/svd

echo "==== job started ===="
date
hostname
echo "PBS_NODEFILE=$PBS_NODEFILE"
cat "$PBS_NODEFILE"

echo "==== choose hybrid nodes ===="
HOST0=$(hostname)
HOST1=$(cat "$PBS_NODEFILE" | sort -u | grep -v "^${HOST0}$" | head -n 1)

if [ -z "$HOST1" ]; then
    HOST1="$HOST0"
fi

printf "%s\n%s\n" "$HOST0" "$HOST1" > ${RUNDIR}/nodes_hybrid

echo "nodes_hybrid:"
cat ${RUNDIR}/nodes_hybrid

echo "==== clean old profiles on master ===="
ssh ${MASTER} "mkdir -p ${MASTER_DIR}/files; rm -f ${MASTER_DIR}/files/mpi_profile*.csv"

echo "==== copy executable to compute nodes ===="
for node in $(cat ${RUNDIR}/nodes_hybrid | sort -u); do
    ssh ${node} "mkdir -p ${RUNDIR}/files; rm -f ${RUNDIR}/files/mpi_profile*.csv ${RUNDIR}/main"
    scp ${MASTER}:${MASTER_DIR}/main ${node}:${RUNDIR}/main
done

cd ${RUNDIR} || exit 1

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "==== runtime config ===="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "main used:"
ls -lh ${RUNDIR}/main

echo "==== run hybrid ===="
timeout 30m /usr/local/bin/mpiexec -np 2 -machinefile ${RUNDIR}/nodes_hybrid ${RUNDIR}/main 20260410
STATUS=$?

echo "run status: $STATUS"

echo "==== profile files on rank0/runtime node ===="
find ${RUNDIR}/files -name "mpi_profile*.csv" -print -exec head -n 2 {} \;

echo "==== copy results back to master ===="
scp -r ${RUNDIR}/files/ ${MASTER}:${MASTER_DIR}/ 2>&1

date
exit $STATUS