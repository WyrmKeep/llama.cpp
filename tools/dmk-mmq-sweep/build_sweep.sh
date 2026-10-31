#!/usr/bin/env bash
# build_sweep.sh <patched llama.cpp tree> <round> <out dir>
#
# Builds the sweep program `llama` (the name that the harness of the GPU host starts) in a CUDA 12.8 builder container:
# one object file for each variant of gen_variants.py, the harness mmq_sweep.cu, and the link.
# The tree is the llama.cpp tree with sweep.patch applied (see README). It is mounted read-only.
# Environment: SWEEP_JOBS (default 8), SWEEP_IMAGE (default localhost/dmk-builder:cuda12.8.1-gcc14).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "--inside" ]]; then
    shift
    round=$1; jobs=${SWEEP_JOBS:-8}
    out=/out
    mkdir -p "$out/obj"
    python3 /sweep/gen_variants.py "$round" "$out"
    flags=(-O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_86,code=[sm_86]" -use_fast_math -extended-lambda
           -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -D_GNU_SOURCE -D_XOPEN_SOURCE=600
           -I/src/ggml/src/ggml-cuda -I/src/ggml/src -I/src/ggml/include -I"$out" -w -Xptxas -v)
    cat > "$out/compile_one.sh" <<EOF
#!/usr/bin/env bash
# compile_one.sh name type nt occ I J streamk fallback [ablation]
set -euo pipefail
name=\$1 type=\$2 nt=\$3 occ=\$4 I=\$5 J=\$6 sk=\$7 fb=\$8 abl=\${9:-0}
extra="\${SWEEP_EXTRA:-}"
[[ \${SWEEP_BASE_NOPIPE:-0} == 1 && \$abl == 0 ]] && extra="\$extra -DGGML_CUDA_MMQ_NO_WEIGHT_PIPE"
[[ -f $out/obj/\$name.o ]] && exit 0
nvcc ${flags[*]} -c /sweep/sweep_variant.cu -o $out/obj/\$name.o \\
  -DSWEEP_NAME=\$name -DSWEEP_TYPE=GGML_TYPE_\$type -DSWEEP_NT=\$nt -DSWEEP_OCC=\$occ -DSWEEP_I=\$I -DSWEEP_J=\$J \\
  -DSWEEP_STREAMK=\$sk -DSWEEP_FALLBACK=\$fb -DSWEEP_ABL=\$abl -DSWEEP_LAYOUT=GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 \$extra \\
  > $out/obj/\$name.log 2>&1 || { echo "FAILED \$name"; exit 1; }
echo "ok \$name"
EOF
    chmod +x "$out/compile_one.sh"
    echo "== compile $(($(wc -l < "$out/variants.tsv") - 1)) variants, -P$jobs"
    t0=$(date +%s)
    tail -n +2 "$out/variants.tsv" | xargs -P "$jobs" -L1 "$out/compile_one.sh" 2>&1 | tee "$out/compile.log" | grep -c '^ok' || true
    grep -h FAILED "$out/compile.log" || true
    t1=$(date +%s)
    echo "== variants done in $((t1 - t0)) s"
    echo "== harness"
    nvcc "${flags[@]}" -c /sweep/mmq_sweep.cu -o "$out/obj/mmq_sweep.o"
    echo "== link"
    objs=$(ls "$out"/obj/*.o)
    nvcc -O3 --generate-code="arch=compute_86,code=[sm_86]" -Xcompiler -static-libstdc++,-static-libgcc $objs -o "$out/llama"
    ls -l "$out/llama"
    exit 0
fi

tree=$(cd "$1" && pwd); round=$2; mkdir -p "$3"; out=$(cd "$3" && pwd)
image=${SWEEP_IMAGE:-localhost/dmk-builder:cuda12.8.1-gcc14}
podman run --rm --security-opt label=disable -v "$tree":/src:ro -v "$here":/sweep:ro -v "$out":/out \
    -e SWEEP_JOBS -e SWEEP_BASE_NOPIPE -e SWEEP_EXTRA "$image" bash /sweep/build_sweep.sh --inside "$round" 2>&1 | tee "$out/build.log"
exit "${PIPESTATUS[0]}"
