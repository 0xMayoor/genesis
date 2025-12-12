#!/usr/bin/env python3
"""
LEVEL 0: FINAL - CLEAN DATA

Fix: Only accept VALID x86-64 mnemonics (filter garbage from objdump)
"""

import os
import subprocess
import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("LEVEL 0: FINAL (Clean Data)")
print("=" * 60)

# Setup
subprocess.run(["apt-get", "update"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "binutils", "gcc", "clang"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "numpy"], capture_output=True)

if "genesis" not in os.getcwd():
    subprocess.run(["rm", "-rf", "genesis"], capture_output=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/0xMayoor/genesis.git"])
    os.chdir("genesis")

print(f"\n[SETUP]")
print(f"  Dir: {os.getcwd()}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ============================================================================
# VALID x86-64 MNEMONICS (Ground Truth)
# ============================================================================

# Complete list of valid x86-64 mnemonics (from Intel/AMD manuals)
VALID_MNEMONICS = {
    # Data transfer
    'mov', 'movabs', 'movzx', 'movsx', 'movsxd', 'movbe', 'movd', 'movq',
    'movs', 'movsb', 'movsw', 'movsd', 'movsq',
    'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmovc', 'cmove', 'cmovg', 'cmovge',
    'cmovl', 'cmovle', 'cmovna', 'cmovnae', 'cmovnb', 'cmovnbe', 'cmovnc',
    'cmovne', 'cmovng', 'cmovnge', 'cmovnl', 'cmovnle', 'cmovno', 'cmovnp',
    'cmovns', 'cmovnz', 'cmovo', 'cmovp', 'cmovpe', 'cmovpo', 'cmovs', 'cmovz',
    'xchg', 'bswap', 'xadd', 'cmpxchg', 'cmpxchg8b', 'cmpxchg16b',
    'push', 'pop', 'pusha', 'pushad', 'popa', 'popad', 'pushf', 'pushfd',
    'pushfq', 'popf', 'popfd', 'popfq',
    'lea', 'lds', 'les', 'lfs', 'lgs', 'lss',
    
    # Arithmetic
    'add', 'adc', 'sub', 'sbb', 'imul', 'mul', 'idiv', 'div',
    'inc', 'dec', 'neg', 'cmp',
    'aaa', 'aas', 'aam', 'aad', 'daa', 'das',
    
    # Logical
    'and', 'or', 'xor', 'not', 'test',
    'sal', 'sar', 'shl', 'shr', 'shld', 'shrd',
    'rol', 'ror', 'rcl', 'rcr',
    
    # Bit operations
    'bt', 'bts', 'btr', 'btc', 'bsf', 'bsr',
    'seta', 'setae', 'setb', 'setbe', 'setc', 'sete', 'setg', 'setge',
    'setl', 'setle', 'setna', 'setnae', 'setnb', 'setnbe', 'setnc', 'setne',
    'setng', 'setnge', 'setnl', 'setnle', 'setno', 'setnp', 'setns', 'setnz',
    'seto', 'setp', 'setpe', 'setpo', 'sets', 'setz',
    'popcnt', 'lzcnt', 'tzcnt',
    
    # Control flow
    'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jae', 'jb', 'jbe', 'jc', 'jnc',
    'jg', 'jge', 'jl', 'jle', 'jo', 'jno', 'jp', 'jpe', 'jpo', 'jnp',
    'js', 'jns', 'jecxz', 'jrcxz',
    'call', 'ret', 'retn', 'retf', 'iret', 'iretd', 'iretq',
    'int', 'int1', 'int3', 'into', 'syscall', 'sysret', 'sysenter', 'sysexit',
    'loop', 'loope', 'loopne', 'loopz', 'loopnz',
    
    # String operations
    'movs', 'cmps', 'scas', 'lods', 'stos',
    'movsb', 'movsw', 'movsd', 'movsq',
    'cmpsb', 'cmpsw', 'cmpsd', 'cmpsq',
    'scasb', 'scasw', 'scasd', 'scasq',
    'lodsb', 'lodsw', 'lodsd', 'lodsq',
    'stosb', 'stosw', 'stosd', 'stosq',
    'rep', 'repe', 'repz', 'repne', 'repnz',
    
    # Flag operations
    'stc', 'clc', 'cmc', 'std', 'cld', 'sti', 'cli',
    'lahf', 'sahf', 'pushf', 'popf',
    
    # Stack frame
    'enter', 'leave',
    
    # Misc
    'nop', 'hlt', 'wait', 'fwait', 'lock', 'xlatb',
    'cpuid', 'rdtsc', 'rdtscp', 'rdpmc', 'rdmsr', 'wrmsr',
    'lfence', 'sfence', 'mfence', 'clflush', 'clflushopt', 'clwb',
    'prefetch', 'prefetchw', 'prefetchnta', 'prefetcht0', 'prefetcht1', 'prefetcht2',
    
    # x87 FPU
    'fld', 'fst', 'fstp', 'fild', 'fist', 'fistp', 'fisttp',
    'fadd', 'faddp', 'fiadd', 'fsub', 'fsubp', 'fisub', 'fsubr', 'fsubrp', 'fisubr',
    'fmul', 'fmulp', 'fimul', 'fdiv', 'fdivp', 'fidiv', 'fdivr', 'fdivrp', 'fidivr',
    'fabs', 'fchs', 'fsqrt', 'fprem', 'fprem1', 'frndint', 'fxtract',
    'fsin', 'fcos', 'fsincos', 'fptan', 'fpatan', 'f2xm1', 'fyl2x', 'fyl2xp1',
    'fcom', 'fcomp', 'fcompp', 'fucom', 'fucomp', 'fucompp', 'ficom', 'ficomp',
    'fcomi', 'fcomip', 'fucomi', 'fucomip',
    'ftst', 'fxam', 'fldz', 'fld1', 'fldpi', 'fldl2t', 'fldl2e', 'fldlg2', 'fldln2',
    'fnop', 'fclex', 'fnclex', 'finit', 'fninit', 'fstcw', 'fnstcw', 'fldcw',
    'fstenv', 'fnstenv', 'fldenv', 'fsave', 'fnsave', 'frstor',
    'fstsw', 'fnstsw', 'ffree', 'ffreep', 'fincstp', 'fdecstp', 'fxch', 'fwait',
    
    # SSE/AVX
    'movss', 'movsd', 'movaps', 'movapd', 'movups', 'movupd',
    'movlps', 'movlpd', 'movhps', 'movhpd', 'movlhps', 'movhlps',
    'movmskps', 'movmskpd', 'movntps', 'movntpd', 'movnti', 'movntq', 'movntdq',
    'movdqa', 'movdqu', 'movq2dq', 'movdq2q',
    'addss', 'addsd', 'addps', 'addpd', 'subss', 'subsd', 'subps', 'subpd',
    'mulss', 'mulsd', 'mulps', 'mulpd', 'divss', 'divsd', 'divps', 'divpd',
    'sqrtss', 'sqrtsd', 'sqrtps', 'sqrtpd', 'rcpss', 'rcpps', 'rsqrtss', 'rsqrtps',
    'maxss', 'maxsd', 'maxps', 'maxpd', 'minss', 'minsd', 'minps', 'minpd',
    'andps', 'andpd', 'andnps', 'andnpd', 'orps', 'orpd', 'xorps', 'xorpd',
    'cmpss', 'cmpsd', 'cmpps', 'cmppd', 'comiss', 'comisd', 'ucomiss', 'ucomisd',
    'cvtsi2ss', 'cvtsi2sd', 'cvtss2si', 'cvtsd2si', 'cvttss2si', 'cvttsd2si',
    'cvtss2sd', 'cvtsd2ss', 'cvtps2pd', 'cvtpd2ps',
    'cvtdq2ps', 'cvtdq2pd', 'cvtps2dq', 'cvtpd2dq', 'cvttps2dq', 'cvttpd2dq',
    'cvtpi2ps', 'cvtpi2pd', 'cvtps2pi', 'cvtpd2pi', 'cvttps2pi', 'cvttpd2pi',
    'punpcklbw', 'punpcklwd', 'punpckldq', 'punpcklqdq',
    'punpckhbw', 'punpckhwd', 'punpckhdq', 'punpckhqdq',
    'packsswb', 'packssdw', 'packuswb', 'packusdw',
    'paddb', 'paddw', 'paddd', 'paddq', 'psubb', 'psubw', 'psubd', 'psubq',
    'paddsb', 'paddsw', 'paddusb', 'paddusw', 'psubsb', 'psubsw', 'psubusb', 'psubusw',
    'pmullw', 'pmulhw', 'pmulhuw', 'pmuludq', 'pmaddwd',
    'pcmpeqb', 'pcmpeqw', 'pcmpeqd', 'pcmpgtb', 'pcmpgtw', 'pcmpgtd',
    'pand', 'pandn', 'por', 'pxor',
    'psllw', 'pslld', 'psllq', 'psrlw', 'psrld', 'psrlq', 'psraw', 'psrad',
    'pslldq', 'psrldq',
    'pshufd', 'pshufhw', 'pshuflw', 'pshufw', 'shufps', 'shufpd',
    'unpcklps', 'unpcklpd', 'unpckhps', 'unpckhpd',
    'pinsrw', 'pextrw', 'pmovmskb', 'maskmovq', 'maskmovdqu',
    'pmaxsw', 'pmaxub', 'pminsw', 'pminub', 'pavgb', 'pavgw', 'psadbw',
    'ldmxcsr', 'stmxcsr',
    
    # Security
    'endbr32', 'endbr64',
    
    # Segment/system (less common but valid)
    'lgdt', 'sgdt', 'lidt', 'sidt', 'lldt', 'sldt', 'ltr', 'str',
    'arpl', 'lar', 'lsl', 'verr', 'verw',
    'clts', 'invd', 'wbinvd', 'invlpg', 'invpcid',
    'in', 'out', 'ins', 'insb', 'insw', 'insd', 'outs', 'outsb', 'outsw', 'outsd',
    'ud0', 'ud1', 'ud2',
    
    # BMI/BMI2
    'andn', 'bextr', 'blsi', 'blsmsk', 'blsr', 'bzhi', 'mulx', 'pdep', 'pext',
    'rorx', 'sarx', 'shlx', 'shrx',
    
    # AES
    'aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc', 'aeskeygenassist',
    'pclmulqdq',
    
    # CRC
    'crc32',
    
    # ADX
    'adcx', 'adox',
    
    # Extensions
    'xsave', 'xrstor', 'xsaveopt', 'xsavec', 'xsaves', 'xrstors', 'xgetbv', 'xsetbv',
    'rdrand', 'rdseed',
    
    # Common with suffixes stripped (objdump sometimes uses these)
    'cltq', 'cqto', 'cwtl', 'cbtw', 'cltd', 'cwtd',
    'cdq', 'cdqe', 'cqo', 'cbw', 'cwde',
    
    # More SSE/AVX
    'ptest', 'pcmpistri', 'pcmpistrm', 'pcmpestri', 'pcmpestrm',
    'pmovzxbw', 'pmovzxbd', 'pmovzxbq', 'pmovzxwd', 'pmovzxwq', 'pmovzxdq',
    'pmovsxbw', 'pmovsxbd', 'pmovsxbq', 'pmovsxwd', 'pmovsxwq', 'pmovsxdq',
    'phaddw', 'phaddd', 'phaddsw', 'phsubw', 'phsubd', 'phsubsw',
    'pmaddubsw', 'pmulhrsw', 'pshufb', 'psignb', 'psignw', 'psignd',
    'palignr', 'pabsb', 'pabsw', 'pabsd',
    'pblendw', 'pblendvb', 'blendps', 'blendpd', 'blendvps', 'blendvpd',
    'dpps', 'dppd', 'extractps', 'insertps',
    'pinsrb', 'pinsrd', 'pinsrq', 'pextrb', 'pextrd', 'pextrq',
    'pmaxsb', 'pmaxsd', 'pmaxuw', 'pmaxud', 'pminsb', 'pminsd', 'pminuw', 'pminud',
    'roundss', 'roundsd', 'roundps', 'roundpd',
    'mpsadbw', 'phminposuw', 'pmuldq', 'pmulld', 'movntdqa', 'pcmpgtq',
    
    # Segment prefixes (sometimes objdump shows these)
    'data16', 'addr32', 'rex', 'fs', 'gs',
    
    # Lock prefix
    'xacquire', 'xrelease',
}

def is_valid_mnemonic(m):
    """Check if mnemonic is valid x86-64."""
    m = m.lower().strip()
    if not m or len(m) > 20:
        return False
    # Direct match
    if m in VALID_MNEMONICS:
        return True
    # Check without suffix (movl -> mov)
    base = re.sub(r'[lqwb]$', '', m)
    if base in VALID_MNEMONICS:
        return True
    # Check for v prefix (AVX)
    if m.startswith('v') and m[1:] in VALID_MNEMONICS:
        return True
    return False

# ============================================================================
# DATA COLLECTION
# ============================================================================
print("\n[DATA COLLECTION]")

def disassemble(binary_path):
    """Disassemble and filter to valid mnemonics only."""
    try:
        result = subprocess.run(
            ["objdump", "-d", "-M", "intel", str(binary_path)],
            capture_output=True, text=True, timeout=60
        )
        if not result.stdout:
            return []
        
        samples = []
        for line in result.stdout.split('\n'):
            match = re.match(r'\s+[0-9a-f]+:\s+([0-9a-f ]+?)\s{2,}(\S+)', line)
            if match:
                bytes_hex = match.group(1).strip().replace(' ', '')
                mnemonic = match.group(2).lower()
                
                # FILTER: Only valid x86 mnemonics
                if not is_valid_mnemonic(mnemonic):
                    continue
                
                # Normalize mnemonic (remove size suffix)
                base = re.sub(r'[lqwb]$', '', mnemonic)
                if base in VALID_MNEMONICS:
                    mnemonic = base
                
                bytes_list = [int(bytes_hex[i:i+2], 16) for i in range(0, len(bytes_hex), 2)]
                if bytes_list:
                    samples.append((bytes_list, mnemonic))
        return samples
    except:
        return []

# Find system binaries
print("  Finding system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/lib/x86_64-linux-gnu"]:
    if Path(path).exists():
        try:
            for f in Path(path).iterdir():
                if f.is_file() and not f.is_symlink():
                    try:
                        with open(f, "rb") as fp:
                            if fp.read(4) == b'\x7fELF':
                                binaries.append(f)
                    except:
                        pass
                if len(binaries) >= 300:
                    break
        except:
            pass
    if len(binaries) >= 300:
        break

print(f"    Found {len(binaries)} system binaries")

# Compile programs
print("  Compiling programs...")
PROGRAMS = {
    "basic": "int main(){return 0;}",
    "loop": "int main(){int s=0;for(int i=0;i<100;i++)s+=i;return s;}",
    "func": "int f(int x){return x*2+1;}int main(){return f(10);}",
    "fptr": "int add(int a,int b){return a+b;}int main(){int(*f)(int,int)=add;return f(3,2);}",
    "indirect": "int f1(){return 1;}int f2(){return 2;}int main(){int(*t[])()={f1,f2};return t[0]()+t[1]();}",
    "tls": "__thread int tls_var=42;int main(){return tls_var;}",
    "ptr": "void swap(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;swap(&x,&y);return x;}",
    "arr": "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return sum(a,5);}",
    "str": "int len(char*s){int n=0;while(s[n])n++;return n;}int main(){return len(\"hello\");}",
    "cond": "int max(int a,int b){return a>b?a:b;}int main(){return max(5,3);}",
    "rec": "int fib(int n){return n<=1?n:fib(n-1)+fib(n-2);}int main(){return fib(10);}",
    "math": "int main(){int a=100,b=37;return ((a+b)*(a-b))/(a%b+1);}",
    "bit": "int main(){int x=0xFF;return (x<<2)|(x>>2)&x^x;}",
    "switch": "int f(int x){switch(x){case 0:return 10;case 1:return 20;case 2:return 30;default:return 0;}}int main(){return f(1);}",
}

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for name, code in PROGRAMS.items():
        for compiler in ["gcc", "clang"]:
            for opt in ["-O0", "-O1", "-O2", "-O3"]:
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)],
                                 capture_output=True, timeout=30).returncode == 0:
                    binaries.append(binary)

print(f"    Total binaries: {len(binaries)}")

# Extract patterns
print("  Extracting instructions...")
all_samples = []
for binary in binaries:
    all_samples.extend(disassemble(binary))

print(f"    Raw samples: {len(all_samples)}")

# Build vocabulary from VALID mnemonics only
mnemonic_counts = defaultdict(int)
for _, mnemonic in all_samples:
    mnemonic_counts[mnemonic] += 1

# Filter by count
valid_mnemonics = {m for m, c in mnemonic_counts.items() if c >= 10}
print(f"    Valid mnemonics: {len(valid_mnemonics)}")

mnemonic_to_id = {m: i for i, m in enumerate(sorted(valid_mnemonics))}
id_to_mnemonic = {i: m for m, i in mnemonic_to_id.items()}
num_classes = len(mnemonic_to_id)

samples = [(b, m) for b, m in all_samples if m in valid_mnemonics]
print(f"    Filtered samples: {len(samples)}")
print(f"    Mnemonics: {list(mnemonic_to_id.keys())[:30]}...")

# ============================================================================
# BALANCE & TRAIN
# ============================================================================
print("\n[BALANCING]")

by_mnemonic = defaultdict(list)
for bytes_list, mnemonic in samples:
    by_mnemonic[mnemonic].append(bytes_list)

MIN_SAMPLES = 50
MAX_SAMPLES = 500

balanced = []
for mnemonic, byte_lists in by_mnemonic.items():
    unique = list(set(tuple(b) for b in byte_lists))
    if len(unique) < MIN_SAMPLES:
        factor = (MIN_SAMPLES // len(unique)) + 1
        unique = unique * factor
    unique = unique[:MAX_SAMPLES]
    for b in unique:
        balanced.append((list(b), mnemonic))

random.shuffle(balanced)
print(f"  Balanced samples: {len(balanced)}")

split = int(0.9 * len(balanced))
train_samples = balanced[:split]
val_samples = balanced[split:]
print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# Dataset
MAX_LEN = 15
PAD_BYTE = 0

class ByteDataset(Dataset):
    def __init__(self, samples, mnemonic_to_id):
        self.samples = samples
        self.mnemonic_to_id = mnemonic_to_id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bytes_list, mnemonic = self.samples[idx]
        if len(bytes_list) < MAX_LEN:
            bytes_list = bytes_list + [0] * (MAX_LEN - len(bytes_list))
        else:
            bytes_list = bytes_list[:MAX_LEN]
        return torch.tensor(bytes_list, dtype=torch.long), torch.tensor(self.mnemonic_to_id[mnemonic], dtype=torch.long)

train_loader = DataLoader(ByteDataset(train_samples, mnemonic_to_id), batch_size=128, shuffle=True)
val_loader = DataLoader(ByteDataset(val_samples, mnemonic_to_id), batch_size=128)

print(f"\n[MODEL]")

class ByteClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(MAX_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(embed_dim, num_classes))
    
    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        emb = self.byte_embed(x) + self.pos_embed(pos)
        enc = self.transformer(emb)
        return self.classifier(enc.mean(dim=1))

model = ByteClassifier(num_classes).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print("\n[TRAINING]")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_val_acc = 0
patience, wait = 30, 0
best_state = None

for epoch in range(500):
    model.train()
    train_correct, train_total = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_correct += (logits.argmax(1) == y).sum().item()
        train_total += y.size(0)
    
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_correct += (model(x).argmax(1) == y).sum().item()
            val_total += y.size(0)
    
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
        if (epoch + 1) % 5 == 0 or val_acc > 0.99:
            print(f"    Epoch {epoch+1}: train={train_acc:.4f} val={val_acc:.4f} *")
    else:
        wait += 1
    
    if wait >= patience and epoch >= 50:
        print(f"    Early stopping at epoch {epoch+1}")
        break
    
    if val_acc >= 0.9999:
        print(f"    Perfect at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.to(device)
print(f"\n  Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# ============================================================================
# GATE TEST
# ============================================================================
print("\n[GATE TEST]")

GATE_PROGRAMS = {
    "reverse": 'void reverse(char*s,int len){for(int i=0;i<len/2;i++){char t=s[i];s[i]=s[len-1-i];s[len-1-i]=t;}}int main(){char s[]="hello";reverse(s,5);return s[0];}',
    "popcount": 'int popcount(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return popcount(0xFF);}',
    "max": 'int max(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5,9};return max(a,6);}',
}

def predict(bytes_list):
    if len(bytes_list) < MAX_LEN:
        bytes_list = bytes_list + [0] * (MAX_LEN - len(bytes_list))
    x = torch.tensor([bytes_list[:MAX_LEN]], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        return id_to_mnemonic[model(x).argmax(1).item()]

correct, total, unknown = 0, 0, 0
errors = []

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for name, code in GATE_PROGRAMS.items():
        for compiler in ["gcc", "clang"]:
            for opt in ["-O0", "-O1", "-O2", "-O3"]:
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)],
                                 capture_output=True).returncode != 0:
                    continue
                
                result = subprocess.run(["objdump", "-d", "-M", "intel", str(binary)],
                                        capture_output=True, text=True)
                
                for line in result.stdout.split('\n'):
                    match = re.match(r'\s+[0-9a-f]+:\s+([0-9a-f ]+?)\s{2,}(\S+)', line)
                    if not match:
                        continue
                    bytes_hex = match.group(1).strip().replace(' ', '')
                    expected = match.group(2).lower()
                    
                    # Normalize expected
                    exp_base = re.sub(r'[lqwb]$', '', expected)
                    if exp_base in VALID_MNEMONICS:
                        expected = exp_base
                    
                    if not is_valid_mnemonic(expected):
                        continue
                    
                    bytes_list = [int(bytes_hex[i:i+2], 16) for i in range(0, len(bytes_hex), 2)]
                    
                    if expected not in mnemonic_to_id:
                        unknown += 1
                        continue
                    
                    pred = predict(bytes_list)
                    total += 1
                    
                    if pred == expected:
                        correct += 1
                    elif len(errors) < 20:
                        errors.append(f"{bytes_hex}: {pred} vs {expected}")

accuracy = 100 * correct / total if total > 0 else 0
print(f"\n  Gate Test: {correct}/{total} = {accuracy:.1f}%")
print(f"  Unknown: {unknown}")

if errors:
    print(f"\n  Errors:")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================================
# SAVE
# ============================================================================
print("\n[SAVE]")

os.makedirs("models/level0_final", exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': id_to_mnemonic,
    'num_classes': num_classes,
    'max_len': MAX_LEN,
}, "models/level0_final/model.pt")

with open("models/level0_final/config.json", "w") as f:
    json.dump({'num_classes': num_classes, 'max_len': MAX_LEN, 
               'mnemonic_to_id': mnemonic_to_id,
               'id_to_mnemonic': {str(k): v for k, v in id_to_mnemonic.items()}}, f, indent=2)

import zipfile
with zipfile.ZipFile("level0_final.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level0_final").iterdir():
        zf.write(f, f"models/level0_final/{f.name}")

print(f"  Created: level0_final.zip")

try:
    import shutil
    os.makedirs("/content/drive/MyDrive/genesis_models", exist_ok=True)
    shutil.copy("level0_final.zip", "/content/drive/MyDrive/genesis_models/")
    print("  Saved to Google Drive")
except:
    pass

try:
    from google.colab import files
    files.download("level0_final.zip")
except:
    print(f"  Download: {os.path.abspath('level0_final.zip')}")

print("\n" + "=" * 60)
print(f"RESULT: {accuracy:.1f}%")
print(f"Classes: {num_classes} (valid x86 mnemonics only)")
print("=" * 60)
