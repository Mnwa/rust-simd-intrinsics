# Manual-vectorization case studies

These are analyses of upstream designs, not copied implementations. No speedup
from an upstream project is attributed to the local Rust examples. Upstream URLs
are listed in [the source index](vectorization-sources.md); branch URLs are discovery
links, not immutable provenance. Before importing code, record SHA, license and
attribution, and recheck memory contracts. No upstream code is vendored here.

## memchr: locate a block, then recover one position [R9]

Problem: a byte-by-byte loop does a comparison and potential branch per byte.
Transformation: compare a block, reduce to a mask, skip all-negative blocks and
recover the first matching lane only when needed. Study boundary loads, early
exit and dispatch as seriously as the comparison instruction.

Local exercise: `arch::find_byte` and portable find/count/common-prefix. Test no
match, match at every lane, early/late match and offsets. Benchmark both latency
and bytes scanned; an early-exit case must not report unscanned bytes as throughput.
Reject a rewrite that changes which match is returned.

## Algorithmica prefix sum: restructure a dependency [R8]

Problem: the accumulator links adjacent iterations. Transformation: prefix scan
inside a block, then propagate a block carry; consider a second pass and cache
blocking to reduce dependency costs without losing memory locality.

Local exercise: SSE2 and portable wrapping-u32 scans. Compare scalar, one-pass and
an optional blocked experiment. Wider-register shifts must account for physical
sub-block boundaries. Include arrays larger than caches. Do not extrapolate the
author's benchmark ratios to the present machine or implementation.

## simdjson: vectorize a stage, not every branch [R10]

Problem: parsing mixes classification, quoting, escaping and structural state.
Transformation: scan chunks into bitmasks, handle quote/escape state, generate
structural indexes, then consume those indexes with a different stage. Prefix XOR
can represent quote parity; boundary state is part of correctness.

Exercise extension: a quoted-delimiter scanner with escape runs crossing every
block boundary. Test backslashes and quotes in the final byte. A simplistic quote
count is not a complete JSON parser or UTF-8 validator. Measure the whole pipeline,
including mask-to-index materialization and the next stage.

## simdutf lookup4: classify combinations rather than branch per byte [R11]

Problem: UTF-8 validity depends on relationships between nearby bytes.
Transformation: encode classes in small lookup tables and combine error flags
across vectors, preserving continuation state at boundaries.

Exercise extension: table-driven byte classification first, then a separately
specified validator. Compare malformed as well as valid input, truncated sequences
and offsets. Do not copy only a table and omit the surrounding state checks.

## Stream VByte: design the representation for the decoder [R16]

Problem: variable byte lengths cause serial parsing and shifts. Transformation:
separate control information from payload; control selects a shuffle pattern for
several integers. The format/layout change is central, not an incidental intrinsic.

Exercise extension: decode one controlled block with a safe slice contract, then
stream blocks. An upstream optimized decoder can require readable padding; do not
transfer that assumption to arbitrary Rust slices. Include the control stream,
format-conversion cost and compression ratio in system-level evaluation.

## zlib-ng Adler-32: algebra removes the apparent recurrence [R12]

Problem: s2 depends on successively updated s1. Transformation: block sum and
weighted block sum, using SAD and multiply-add building blocks in optimized ISA
paths, with bounded accumulation before modular reduction.

Local exercise: `portable::adler32`; the derivation and its u32 bound are in
[patterns](vectorization-patterns.md). Benchmark realistic byte distributions and
large inputs. Modulo frequency is a performance parameter only after proving the
intermediate range safe.

## wide reducers: combine halves instead of spilling every lane [R5]

Problem: the public operation may have no one-instruction implementation.
Transformation: combine vector halves and finish a short tree. Compare to a lane
array epilogue; neither source form alone proves the generated result.

Local exercise: Fearless fixed-width sum/product versus native-width epilogue and
1/2/4 accumulators. Keep modulo semantics identical. Generic and native builds may
compile `wide`'s cfg-controlled representation differently.

## Arm vector math / SLEEF: a polynomial is not the whole function [R13,R14]

Problem: log/exp are not basic SIMD instructions on the target. Transformation:
range reduction, approximating a small interval, reconstruction and explicit
special cases. Accuracy requirements drive coefficient and evaluation choices.

Local exercise: bounded-domain u32 logarithm and histogram score, not a copied
production log2. Test the whole expression and its threshold use, not only the
polynomial at a few convenient points.

## Fearless sRGB: one specialized helper inside a portable kernel [R15]

Problem: a small packing/blending operation can benefit from target-specific
instructions while the surrounding math is portable. Transformation: keep the
main algorithm generic and specialize a primitive through the library's kernel
mechanism. Inspect the actual version's `kernel!` syntax before writing it.

Exercise extension: specialize the local reducer or packing operation, leaving
dispatch, scalar oracle and tails unchanged. Include a correct fallback. The
existence of a specialized instruction is not an end-to-end performance result.
