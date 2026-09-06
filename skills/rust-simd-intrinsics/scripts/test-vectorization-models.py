#!/usr/bin/env python3
"""Independent scalar models of the algebra. NOT Rust compilation/ISA validation."""
from __future__ import annotations
import itertools
import math
import random
import unittest
import zlib
MASK=(1<<32)-1

def rotate(v,k):
    return v[k:]+v[:k]
def tree4(values,op):
    p=[op(a,b)&MASK for a,b in zip(values,rotate(values,2))]
    return [op(a,b)&MASK for a,b in zip(p,rotate(p,1))][0]
def scan4(values):
    carry=0; out=[]
    for base in range(0,len(values)//4*4,4):
        v=list(values[base:base+4])
        for shift in (1,2):
            shifted=[0]*shift+v[:-shift]
            v=[(a+b)&MASK for a,b in zip(v,shifted)]
        v=[(x+carry)&MASK for x in v]
        carry=v[3]; out.extend(v)
    for x in values[len(out):]:
        carry=(carry+x)&MASK; out.append(carry)
    return out

def log_count(n):
    n=max(1,n); e=n.bit_length()-1; m=n/(1<<e)
    z=(m-1)/(m+1); q=z*z; p=1/19
    for k in reversed(range(9)):
        p=p*q+1/(2*k+1)
    return e+2/math.log(2)*z*p

class AlgebraTests(unittest.TestCase):
    def test_wrapping_reduction_trees(self):
        rng=random.Random(721)
        for _ in range(20000):
            x=[rng.getrandbits(32) for _ in range(4)]
            self.assertEqual(tree4(x,lambda a,b:a+b),sum(x)&MASK)
            self.assertEqual(tree4(x,lambda a,b:a*b),math.prod(x)&MASK)
    def test_native_accumulator_organization(self):
        rng=random.Random(91)
        for width,accumulators in itertools.product([1,4,8,16],[1,2,4]):
            for n in range(258):
                values=[rng.getrandbits(32) for _ in range(n)]
                acc=[[0]*width for _ in range(accumulators)]
                full=n//(width*accumulators)*(width*accumulators)
                for base in range(0,full,width*accumulators):
                    for a in range(accumulators):
                        for lane in range(width):
                            acc[a][lane]=(acc[a][lane]+values[base+a*width+lane])&MASK
                cursor=full
                while cursor+width<=n:
                    for lane in range(width):
                        acc[0][lane]=(acc[0][lane]+values[cursor+lane])&MASK
                    cursor+=width
                result=(sum(map(sum,acc))+sum(values[cursor:]))&MASK
                self.assertEqual(result,sum(values)&MASK)
    def test_scan(self):
        rng=random.Random(818)
        for n in range(258):
            x=[rng.getrandbits(32) for _ in range(n)]
            carry=0; expected=[]
            for value in x:
                carry=(carry+value)&MASK; expected.append(carry)
            self.assertEqual(scan4(x),expected)
    def test_sorting_network(self):
        def stage(v,permutation,low):
            other=[v[i] for i in permutation]
            return [min(a,b) if select else max(a,b) for a,b,select in zip(v,other,low)]
        for original in itertools.product(range(6),repeat=4):
            v=stage(list(original),[1,0,3,2],[1,0,1,0])
            v=stage(v,[2,3,0,1],[1,1,0,0])
            v=stage(v,[0,2,1,3],[1,1,0,0])
            self.assertEqual(v,sorted(original))
    def test_block_adler_algebra(self):
        rng=random.Random(312)
        for n in list(range(258))+[4097,65537]:
            data=bytes(rng.randrange(256) for _ in range(n))
            a,b=1,0
            for base in range(0,n//16*16,16):
                x=data[base:base+16]
                b=(b+16*a+sum((16-i)*value for i,value in enumerate(x)))%65521
                a=(a+sum(x))%65521
            for x in data[n//16*16:]:
                a=(a+x)%65521; b=(b+a)%65521
            self.assertEqual((b<<16)|a,zlib.adler32(data))
    def test_byte_swar_popcount(self):
        for byte in range(256):
            x=byte-((byte>>1)&0x55)
            x=(x&0x33)+((x>>2)&0x33)
            x=(x+(x>>4))&0x0f
            self.assertEqual(x,byte.bit_count())
    def test_log_series_sampled_domain(self):
        rng=random.Random(117)
        samples=list(range(1,65537))+[rng.randrange(1,1<<32) for _ in range(100000)]
        errors=[abs(log_count(n)-math.log2(n)) for n in samples]
        self.assertLessEqual(max(errors),1e-9)
        print(f'Python model: sampled max log2 error={max(errors):.17g}; this is not a Rust/backend test.')
    def test_histogram_scatter_counterexample(self):
        histogram=[0]*256
        indices=[5,5]
        gathered=[histogram[i] for i in indices]
        for i,old in zip(indices,gathered):
            histogram[i]=old+1
        self.assertEqual(histogram[5],1)  # Incorrect naive SIMD update, intentionally.
        expected=[0]*256
        for i in indices:
            expected[i]+=1
        self.assertEqual(expected[5],2)
        self.assertNotEqual(histogram,expected)

if __name__=='__main__':
    unittest.main()
