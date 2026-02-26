#!/usr/bin/env python3
"""
Key: "LOVECSND"
Ciphertext: 1100101011101101101000100110010101011111101101110011100001110011
"""

# DES Tables

PC1 = [57,49,41,33,25,17, 9,
        1,58,50,42,34,26,18,
       10, 2,59,51,43,35,27,
       19,11, 3,60,52,44,36,
       63,55,47,39,31,23,15,
        7,62,54,46,38,30,22,
       14, 6,61,53,45,37,29,
       21,13, 5,28,20,12, 4]

PC2 = [14,17,11,24, 1, 5,
        3,28,15, 6,21,10,
       23,19,12, 4,26, 8,
       16, 7,27,20,13, 2,
       41,52,31,37,47,55,
       30,40,51,45,33,48,
       44,49,39,56,34,53,
       46,42,50,36,29,32]

SHIFTS = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

IP = [58,50,42,34,26,18,10, 2,
      60,52,44,36,28,20,12, 4,
      62,54,46,38,30,22,14, 6,
      64,56,48,40,32,24,16, 8,
      57,49,41,33,25,17, 9, 1,
      59,51,43,35,27,19,11, 3,
      61,53,45,37,29,21,13, 5,
      63,55,47,39,31,23,15, 7]

IP_INV = [40, 8,48,16,56,24,64,32,
          39, 7,47,15,55,23,63,31,
          38, 6,46,14,54,22,62,30,
          37, 5,45,13,53,21,61,29,
          36, 4,44,12,52,20,60,28,
          35, 3,43,11,51,19,59,27,
          34, 2,42,10,50,18,58,26,
          33, 1,41, 9,49,17,57,25]

E_TABLE = [32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9,10,11,12,13,
           12,13,14,15,16,17,
           16,17,18,19,20,21,
           20,21,22,23,24,25,
           24,25,26,27,28,29,
           28,29,30,31,32, 1]

P_BOX = [16, 7,20,21,
         29,12,28,17,
          1,15,23,26,
          5,18,31,10,
          2, 8,24,14,
         32,27, 3, 9,
         19,13,30, 6,
         22,11, 4,25]

S_BOXES = [
    # S1
    [[14, 4,13, 1, 2,15,11, 8, 3,10, 6,12, 5, 9, 0, 7],
     [ 0,15, 7, 4,14, 2,13, 1,10, 6,12,11, 9, 5, 3, 8],
     [ 4, 1,14, 8,13, 6, 2,11,15,12, 9, 7, 3,10, 5, 0],
     [15,12, 8, 2, 4, 9, 1, 7, 5,11, 3,14,10, 0, 6,13]],
    # S2
    [[15, 1, 8,14, 6,11, 3, 4, 9, 7, 2,13,12, 0, 5,10],
     [ 3,13, 4, 7,15, 2, 8,14,12, 0, 1,10, 6, 9,11, 5],
     [ 0,14, 7,11,10, 4,13, 1, 5, 8,12, 6, 9, 3, 2,15],
     [13, 8,10, 1, 3,15, 4, 2,11, 6, 7,12, 0, 5,14, 9]],
    # S3
    [[10, 0, 9,14, 6, 3,15, 5, 1,13,12, 7,11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6,10, 2, 8, 5,14,12,11,15, 1],
     [13, 6, 4, 9, 8,15, 3, 0,11, 1, 2,12, 5,10,14, 7],
     [ 1,10,13, 0, 6, 9, 8, 7, 4,15,14, 3,11, 5, 2,12]],
    # S4
    [[ 7,13,14, 3, 0, 6, 9,10, 1, 2, 8, 5,11,12, 4,15],
     [13, 8,11, 5, 6,15, 0, 3, 4, 7, 2,12, 1,10,14, 9],
     [10, 6, 9, 0,12,11, 7,13,15, 1, 3,14, 5, 2, 8, 4],
     [ 3,15, 0, 6,10, 1,13, 8, 9, 4, 5,11,12, 7, 2,14]],
    # S5
    [[ 2,12, 4, 1, 7,10,11, 6, 8, 5, 3,15,13, 0,14, 9],
     [14,11, 2,12, 4, 7,13, 1, 5, 0,15,10, 3, 9, 8, 6],
     [ 4, 2, 1,11,10,13, 7, 8,15, 9,12, 5, 6, 3, 0,14],
     [11, 8,12, 7, 1,14, 2,13, 6,15, 0, 9,10, 4, 5, 3]],
    # S6
    [[12, 1,10,15, 9, 2, 6, 8, 0,13, 3, 4,14, 7, 5,11],
     [10,15, 4, 2, 7,12, 9, 5, 6, 1,13,14, 0,11, 3, 8],
     [ 9,14,15, 5, 2, 8,12, 3, 7, 0, 4,10, 1,13,11, 6],
     [ 4, 3, 2,12, 9, 5,15,10,11,14, 1, 7, 6, 0, 8,13]],
    # S7
    [[ 4,11, 2,14,15, 0, 8,13, 3,12, 9, 7, 5,10, 6, 1],
     [13, 0,11, 7, 4, 9, 1,10,14, 3, 5,12, 2,15, 8, 6],
     [ 1, 4,11,13,12, 3, 7,14,10,15, 6, 8, 0, 5, 9, 2],
     [ 6,11,13, 8, 1, 4,10, 7, 9, 5, 0,15,14, 2, 3,12]],
    # S8
    [[13, 2, 8, 4, 6,15,11, 1,10, 9, 3,14, 5, 0,12, 7],
     [ 1,15,13, 8,10, 3, 7, 4,12, 5, 6,11, 0,14, 9, 2],
     [ 7,11, 4, 1, 9,12,14, 2, 0, 6,10,13,15, 3, 5, 8],
     [ 2, 1,14, 7, 4,10, 8,13,15,12, 9, 0, 3, 5, 6,11]],
]

# Helper Functions
def permute(block, table):
    return [block[t-1] for t in table]

def left_shift(bits, n):
    return bits[n:] + bits[:n]

def xor(a, b):
    return [x ^ y for x, y in zip(a, b)]

def bits_to_str(bits):
    return ''.join(str(b) for b in bits)

def bits_to_hex(bits):
    result = ''
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        result += hex(int(''.join(str(b) for b in nibble), 2))[2:].upper()
    return result

def bits_to_char(bits):
    chars = ''
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        val = int(''.join(str(b) for b in byte), 2)
        chars += chr(val) if 32 <= val < 127 else f'[{val}]'
    return chars

def format_bits(bits, group=4):
    s = bits_to_str(bits)
    return ' '.join(s[i:i+group] for i in range(0, len(s), group))

# Key Schedule

def generate_subkeys(key_bits):
    # Generate 16 48-bit subkeys from 64-bit key
    kp = permute(key_bits, PC1)          # 56 bits after PC-1
    C, D = kp[:28], kp[28:]
    subkeys = []
    Cs, Ds = [C[:]], [D[:]]
    for i in range(16):
        C = left_shift(C, SHIFTS[i])
        D = left_shift(D, SHIFTS[i])
        Cs.append(C[:])
        Ds.append(D[:])
        subkeys.append(permute(C + D, PC2))
    return subkeys, Cs, Ds

# f Function

def f_function(R, K):
    # The Feistel f function
    # Expand R from 32 to 48 bits
    ER = permute(R, E_TABLE)
    # XOR with subkey
    xored = xor(ER, K)
    # S-box substitution
    sbox_out = []
    for i in range(8):
        group = xored[i*6:(i+1)*6]
        row = (group[0] << 1) | group[5]
        col = (group[1] << 3) | (group[2] << 2) | (group[3] << 1) | group[4]
        val = S_BOXES[i][row][col]
        sbox_out.extend([(val >> (3-j)) & 1 for j in range(4)])
    # P permutation
    return permute(sbox_out, P_BOX), ER, xored, sbox_out

# DES Core (Encryption or Decryption via key order)

def des_process(block_bits, subkeys, label=""):
    # Run 16 DES rounds. For decryption pass reversed subkeys
    # Initial permutation
    ip_out = permute(block_bits, IP)
    L, R = ip_out[:32], ip_out[32:]

    print(f"\n{'─'*70}")
    print(f"  Initial Permutation (IP)")
    print(f"  IP output : {format_bits(ip_out)}")
    print(f"  L0        : {format_bits(L)}")
    print(f"  R0        : {format_bits(R)}")
    print(f"{'─'*70}")

    round_data = []
    for n in range(1, 17):
        K = subkeys[n-1]
        f_out, ER, xored, sbox_out = f_function(R, K)
        new_R = xor(L, f_out)
        new_L = R[:]

        print(f"\n  ── Round {n:2d} ──")
        print(f"  K{n:<2d}        : {format_bits(K, 6)}")
        print(f"  E(R{n-1:<2d})     : {format_bits(ER, 6)}")
        print(f"  K⊕E(R)    : {format_bits(xored, 6)}")
        print(f"  S-box out : {format_bits(sbox_out)}")
        print(f"  f(R{n-1},K{n:<2d})  : {format_bits(f_out)}")
        print(f"  L{n:<2d}        : {format_bits(new_L)}")
        print(f"  R{n:<2d}        : {format_bits(new_R)}")

        round_data.append((n, new_L[:], new_R[:], f_out[:], K[:]))
        L, R = new_L, new_R

    # Pre-output: swap L and R
    preoutput = R + L
    output = permute(preoutput, IP_INV)

    print(f"\n{'─'*70}")
    print(f"  Pre-output (R16L16): {format_bits(preoutput)}")
    print(f"  Final IP⁻¹ output  : {format_bits(output)}")
    print(f"{'─'*70}")

    return output, round_data

# Main

def main():
    # Inputs
    ciphertext_str = ("1100101011101101101000100110010101011111101101110011100001110011")
    key_str        = ("0100110001001111010101100100010101000011010100110100111001000100")

    ciphertext_bits = [int(b) for b in ciphertext_str]
    key_bits        = [int(b) for b in key_str]

    print(f"\n  Key (ASCII)  : LOVECSND")
    print(f"  Key (binary) : {format_bits(key_bits)}")
    print(f"  Ciphertext   : {format_bits(ciphertext_bits)}")

    # Step 1: Generate subkeys
    print("\n" + "=" * 70)
    print("  STEP 1 — GENERATE 16 ROUND KEYS")
    print("=" * 70)

    subkeys, Cs, Ds = generate_subkeys(key_bits)

    kp = permute(key_bits, PC1)
    print(f"\n  PC-1 output: {format_bits(kp)}")
    print(f"  C0 : {bits_to_str(Cs[0])}")
    print(f"  D0 : {bits_to_str(Ds[0])}")
    print()
    for i in range(1, 17):
        print(f"  C{i:<2d}: {bits_to_str(Cs[i])}   D{i:<2d}: {bits_to_str(Ds[i])}")

    print("\n  Generated Round Keys:")
    print()
    for i, sk in enumerate(subkeys, 1):
        print(f"  K{i:<2d}: {format_bits(sk, 6)}")

    # Step 2: Decryption
    print("\n" + "=" * 70)
    print("  STEP 2 — DES DECRYPTION)")
    print("=" * 70)

    reversed_subkeys = list(reversed(subkeys))
    plaintext_bits, round_data = des_process(ciphertext_bits, reversed_subkeys)

    # Summary Table
    print("\n" + "=" * 70)
    print("  SUMMARY — Round Keys, f outputs, and LnRn")
    print("=" * 70)
    print(f"  {'Round':<6} {'Kn (used)':<50} {'f(Rn-1,Kn)':<34} {'Ln':34} {'Rn'}")
    print(f"  {'─'*6} {'─'*50} {'─'*34} {'─'*34} {'─'*34}")
    for n, L, R, f_out, K in round_data:
        print(f"  {n:<6} {bits_to_str(K):<50} {bits_to_str(f_out):<34} {bits_to_str(L):<34} {bits_to_str(R)}")

    # Result
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"\n  Plaintext (binary): {format_bits(plaintext_bits)}")
    print(f"  Plaintext (hex)   : {bits_to_hex(plaintext_bits)}")
    print(f"  Plaintext (ASCII) : {bits_to_char(plaintext_bits)}")
    print()

if __name__ == "__main__":
    main()