#!/usr/bin/env python3
"""Make variants.inc and variants.tsv for the MMQ tile sweep.

A variant is (type, threads for each block, occupancy of __launch_bounds__, I, J, stream-k, fallback).
The row tile height I is 16 times the number of warps (the MMA code of the kernel needs it).

Usage: gen_variants.py <round> <out dir>
Rounds:
  r1   the first sweep: three block shapes, J from 8 to 128, Q4_0 and Q8_0, no fallback
  t0   a trial of three variants for the build script
  fb   the fallback table (a number of weight rows that is not a multiple of 128): J of 8, 16, 32, 64 and 128
  ab   the baseline table (256 threads, one block for each SM, I of 128, J from 8 to 128) and, for each J, six ablations of the loads
       of the k loop (SWEEP_ABL 1: no weight load, 2: no activation load, 3: neither, 4: weight loads read the first k block again,
       8: the same for the activation loads, 12: both). The ablations are timing only, the result is wrong.
  pipe the weight tile pipeline of change 0009 (Q4_0 only): the baseline variants of the table (256 threads, one block for each SM, I of 128,
       J from 8 to 128) are built without the pipeline (build_sweep.sh with SWEEP_BASE_NOPIPE=1) and the variants with suffix _abl1 are built with it.
       Both give the same result, so ndiff against the baseline checks the bit identity of the pipeline.
  pipec  three variants of the pipeline (J of 64, 120 and 128) to read the registers and the spills of the build log. The environment variable
       SWEEP_EXTRA of build_sweep.sh adds flags for nvcc to each variant.
"""
import sys
import os

R = sys.argv[1]
out = sys.argv[2]
os.makedirs(out, exist_ok=True)

J_ALL = list(range(8, 129, 8))

def configs_r1():
    # (nthreads, I, occupancy, list of J)
    yield (256, 128, 1, J_ALL)                      # the baseline table of the build
    yield (256, 128, 2, list(range(8, 73, 8)))      # two blocks for each SM: 128 registers, 47 KB shared memory at J 64
    yield (128, 64, 1, J_ALL)                       # half tile, 255 registers allowed
    yield (128, 64, 3, list(range(8, 81, 8)))       # 170 registers
    yield (128, 64, 4, list(range(8, 41, 8)))       # 128 registers
    yield (512, 256, 1, J_ALL[1:])                  # double tile, 128 registers (J 8 does not compile: 16 warps need J >= 16)

def configs_fb():
    for nt, I, occ in ((256, 128, 1), (128, 64, 1), (64, 32, 1), (64, 32, 4), (128, 64, 3)):
        yield (nt, I, occ, [8, 16, 32, 64, 128])

def configs_t0():
    yield (256, 128, 1, [64])
    yield (256, 128, 2, [64])
    yield (128, 64, 3, [64])

def configs_ab():
    yield (256, 128, 1, J_ALL)

def configs_pipe():
    yield (256, 128, 1, J_ALL)

def configs_pipec():
    yield (256, 128, 1, [64, 120, 128])

gens = {"r1": configs_r1, "fb": configs_fb, "t0": configs_t0, "ab": configs_ab, "pipe": configs_pipe, "pipec": configs_pipec}
fallbacks = {"r1": [0], "fb": [1], "t0": [0], "ab": [0], "pipe": [0], "pipec": [0]}
ablations = {"r1": [0], "fb": [0], "t0": [0], "ab": [0, 1, 2, 3, 4, 8, 12], "pipe": [0, 1], "pipec": [1]}
types = {"r1": ("Q4_0", "Q8_0"), "fb": ("Q4_0", "Q8_0"), "t0": ("Q4_0", "Q8_0"), "ab": ("Q4_0", "Q8_0"), "pipe": ("Q4_0",), "pipec": ("Q4_0",)}

rows = []
for nt, I, occ, js in gens[R]():
    assert I == 16 * (nt // 32), (nt, I)
    for t in types[R]:
        for J in js:
            for sk in (1,):
                for fb in fallbacks[R]:
                    for abl in ablations[R]:
                        name = f"{t[:2].lower()}{t[3]}_nt{nt}_o{occ}_I{I}_J{J}_s{sk}_f{fb}" + (f"_abl{abl}" if abl else "")
                        rows.append((name, t, nt, occ, I, J, sk, fb, abl))

with open(os.path.join(out, "variants.inc"), "w") as f:
    for name, t, nt, occ, I, J, sk, fb, abl in rows:
        f.write(f"VARIANT({name}, GGML_TYPE_{t}, {nt}, {occ}, {I}, {J}, {sk}, {fb})\n")
with open(os.path.join(out, "variants.tsv"), "w") as f:
    f.write("name\ttype\tnt\tocc\tI\tJ\tstreamk\tfallback\tablation\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(f"{len(rows)} variants")
