import os
import gzip
import math
import pickle
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from math import exp, log
from copy import deepcopy
from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import PandasTools, Crippen, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
rdBase.DisableLog('rdApp.error')
# ============================================================================
# Build the vocabulary for SMILES. Besides, definite vectorize function (atoms -> numerics) and devectorize function (numerics -> atoms)
class Tokenizer():

    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = " "
    
    def build_vocab(self):
        chars=[]
        # atoms (carbon), replace Cl for Q and Br for W
        chars = chars + ['H', 'B', 'c', 'C', 'n', 'N', 'o', 'O', 'p', 'P', 's', 'S', 'F', 'Q', 'W', 'I']
        # hidrogens: H2 to Z, H3 to X
        chars = chars + ['[', ']', '+', 'Z', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['1', '2', '3', '4', '5', '6', '7']
        # anit/clockwise
        chars = chars + ['@']
        # directional bonds
        chars = chars + ['/', '\\']
        #Important that pad gets value 0
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)
            
    @property
    def tokenlist(self):
        return self._tokenlist
    
    @tokenlist.setter
    def tokenlist(self, tokenlist):
        self._tokenlist = tokenlist
        # create the dictionaries      
        self.char_to_int = {c:i for i,c in enumerate(self._tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}
    
    def encode(self, smi):
        encoded = []
        smi = smi.replace('Cl', 'Q')
        smi = smi.replace('Br', 'W')
        # hydrogens
        smi = smi.replace('H2', 'Z')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.end]]
    
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords]) 
        # hydrogens
        smi = smi.replace('Z', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Cl')
        smi = smi.replace('W', 'Br')
        
        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)


# ============================================================================
# Select chemical properties
def reward_fn(properties, generated_smiles):
    if properties == 'druglikeness':
        vals = batch_druglikeness(generated_smiles) 
    elif properties == 'solubility':
        vals = batch_solubility(generated_smiles)
    elif properties == 'synthesizability':
        vals = batch_SA(generated_smiles)   
    return vals

# Diversity
def batch_diversity(smiles):
    scores = []
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in df['mol'] if m is not None]
    for i in range(1, len(fps)):
        scores.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True))
    return np.mean(scores)

# ============================================================================
# Druglikeness
AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')
AcceptorSmarts = [
    '[oH0;X2]',
    '[OH1;X2;v2]',
    '[OH0;X2;v2]',
    '[OH0;X1;v2]',
    '[O-;X1]',
    '[SH0;X2;v2]',
    '[SH0;X1;v2]',
    '[S-;X1]',
    '[nH0;X2]',
    '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]']
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))
StructuralAlertSmarts = [
    '*1[O,S,N]*1',
    '[S,C](=[O,S])[F,Br,Cl,I]',
    '[CX4][Cl,Br,I]',
    '[C,c]S(=O)(=O)O[C,c]',
    '[$([CH]),$(CC)]#CC(=O)[C,c]',
    '[$([CH]),$(CC)]#CC(=O)O[C,c]',
    'n[OH]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
    'C=C(C=O)C=O',
    'n1c([F,Cl,Br,I])cccc1',
    '[CH1](=O)',
    '[O,o][O,o]',
    '[C;!R]=[N;!R]',
    '[N!R]=[N!R]',
    '[#6](=O)[#6](=O)',
    '[S,s][S,s]',
    '[N,n][NH2]',
    'C(=O)N[NH2]',
    '[C,c]=S',
    '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
    'C1(=[O,N])C=CC(=[O,N])C=C1',
    'C1(=[O,N])C(=[O,N])C=CC=C1',
    'a21aa3a(aa1aaaa2)aaaa3',
    'a31a(a2a(aa1)aaaa2)aaaa3',
    'a1aa2a3a(a1)A=AA=A3=AA=A2',
    'c1cc([NH2])ccc1',
    '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
    'I',
    'OS(=O)(=O)[O-]',
    '[N+](=O)[O-]',
    'C(=O)N[OH]',
    'C1NC(=O)NC(=O)1',
    '[SH]',
    '[S-]',
    'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
    'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
    '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
    '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
    '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
    '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    'C#C',
    '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
    '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
    '[C,c]=N[OH]',
    '[C,c]=NOC=O',
    '[C,c](=O)[CX4,CR0X3,O][C,c](=O)',
    'c1ccc2c(c1)ccc(=O)o2',
    '[O+,o+,S+,s+]',
    'N=C=O',
    '[NX3,NX4][F,Cl,Br,I]',
    'c1ccccc1OC(=O)[#6]',
    '[CR0]=[CR0][CR0]=[CR0]',
    '[C+,c+,C-,c-]',
    'N=[N+]=[N-]',
    'C12C(NC(N1)=O)CSC2',
    'c1c([OH])c([OH,NH2,NH])ccc1',
    'P',
    '[N,O,S]C#N',
    'C=C=O',
    '[Si][F,Cl,Br,I]',
    '[SX2]O',
    '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
    'O1CCCCC1OC2CCC3CCCCC3C2',
    'N=[CR0][N,n,O,S]',
    '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
    'C=[C!r]C#N',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
    '[OH]c1ccc([OH,NH2,NH])cc1',
    'c1ccccc1OC(=O)O',
    '[SX2H0][N]',
    'c12ccccc1(SC(S)=N2)',
    'c12ccccc1(SC(=S)N2)',
    'c1nnnn1C=O',
    's1c(S)nnc1NC=O',
    'S1C=CSC1=S',
    'C(=O)Onnn',
    'OS(=O)(=O)C(F)(F)F',
    'N#CC[OH]',
    'N#CC(=O)',
    'S(=O)(=O)C#N',
    'N[CH2]C#N',
    'C1(=O)NCC1',
    'S(=O)(=O)[O-,OH]',
    'NC[F,Cl,Br,I]',
    'C=[C!r]O',
    '[NX2+0]=[O+0]',
    '[OR0,NR0][OR0,NR0]',
    'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
    '[CX2R0][NX3R0]',
    'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
    '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
    '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
    '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
    '[*]=[N+]=[*]',
    '[SX3](=O)[O-,OH]',
    'N#N',
    'F.F.F.F',
    '[R0;D2][R0;D2][R0;D2][R0;D2]',
    '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
    'C=!@CC=[O,S]',
    '[#6,#8,#16][C,c](=O)O[C,c]',
    'c[C;R0](=[O,S])[C,c]',
    'c[SX2][C;!R]',
    'C=C=C',
    'c1nc([F,Cl,Br,I,S])ncc1',
    'c1ncnc([F,Cl,Br,I,S])c1',
    'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
    '[C,c]S(=O)(=O)c1ccc(cc1)F',
    '[15N]',
    '[13C]',
    '[18O]',
    '[34S]']
StructuralAlerts = []
for smarts in StructuralAlertSmarts:
    StructuralAlerts.append(Chem.MolFromSmarts(smarts))
pads1 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [0.486849448, 186.2293718, 2.066177165, 3.902720615, 1.027025453, 0.913012565, 145.4314800],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]
pads2 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]

def ads(x, a, b, c, d, e, f, dmax):
    return ((a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f))))) / dmax)

def qed_eval(w, p, gerebtzoff):
    d = [0.00] * 8
    if gerebtzoff:
        for i in range(0, 8):
            d[i] = ads(p[i], pads1[i][0], pads1[i][1], pads1[i][2], pads1[i][3], pads1[i][4], pads1[i][5], pads1[i][6])
    else:
        for i in range(0, 8):
            d[i] = ads(p[i], pads2[i][0], pads2[i][1], pads2[i][2], pads2[i][3], pads2[i][4], pads2[i][5], pads2[i][6])
    t = 0.0
    for i in range(0, 8):
        t += w[i] * log(d[i])
    return (exp(t / sum(w)))

def qed(mol):
    matches = []
    if mol is None:
        raise WrongArgument("properties(mol)", "mol argument is \'None\'")
    x = [0] * 9
    x[0] = Descriptors.MolWt(mol)
    x[1] = Descriptors.MolLogP(mol)
    for hba in Acceptors:
        if mol.HasSubstructMatch(hba):
            matches = mol.GetSubstructMatches(hba)
            x[2] += len(matches)
    x[3] = Descriptors.NumHDonors(mol)          
    x[4] = Descriptors.TPSA(mol)
    x[5] = Descriptors.NumRotatableBonds(mol)
    x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(deepcopy(mol), AliphaticRings))
    for alert in StructuralAlerts:
        if (mol.HasSubstructMatch(alert)):
            x[7] += 1
    ro5_failed = 0
    if x[3] > 5:
        ro5_failed += 1
    if x[2] > 10:
        ro5_failed += 1
    if x[0] >= 500:
        ro5_failed += 1
    if x[1] > 5:
        ro5_failed += 1
    x[8] = ro5_failed
    return qed_eval([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], x, True)

def batch_druglikeness(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            vals.append(0.0)
        else:
            val = qed(mol)
            vals.append(val)
    return vals

# Solubility
def batch_solubility(smiles):
    vals = []
    for sm in smiles: 
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            vals.append(0.0)
        else:
            low_logp = -2.12178879609
            high_logp = 6.0429063424
            logp = Crippen.MolLogP(mol)
            val = (logp - low_logp) / (high_logp - low_logp)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
    return vals

# Read synthesizability model
def readSAModel(filename='SA_score.pkl.gz'):
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return SA_model
SA_model = readSAModel()

#synthesizability
def batch_SA(smiles):
    vals = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if sm != '' and mol is not None and mol.GetNumAtoms() > 1:
            # fragment score
            fp = Chem.AllChem.GetMorganFingerprint(mol, 2)
            fps = fp.GetNonzeroElements()
            score1 = 0.
            nf = 0
            for bitId, v in fps.items():
                nf += v
                sfp = bitId
                score1 += SA_model.get(sfp, -4) * v
            score1 /= nf
            # features score
            nAtoms = mol.GetNumAtoms()
            nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            ri = mol.GetRingInfo()
            nSpiro = Chem.AllChem.CalcNumSpiroAtoms(mol)
            nBridgeheads = Chem.AllChem.CalcNumBridgeheadAtoms(mol)
            nMacrocycles = 0
            for x in ri.AtomRings():
                if len(x) > 8:
                    nMacrocycles += 1
            sizePenalty = nAtoms**1.005 - nAtoms
            stereoPenalty = math.log10(nChiralCenters + 1)
            spiroPenalty = math.log10(nSpiro + 1)
            bridgePenalty = math.log10(nBridgeheads + 1)
            macrocyclePenalty = 0.
            if nMacrocycles > 0:
                macrocyclePenalty = math.log10(2)
            score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
            score3 = 0.
            if nAtoms > len(fps):
                score3 = math.log(float(nAtoms) / len(fps)) * .5
            sascore = score1 + score2 + score3
            min = -4.0
            max = 2.5
            sascore = 11. - (sascore - min + 1) / (max - min) * 9.
            # smooth the 10-end
            if sascore > 8.:
                sascore = 8. + math.log(sascore + 1. - 9.)
            if sascore > 10.:
                sascore = 10.0
            elif sascore < 1.:
                sascore = 1.0
            val = (sascore - 5) / (1.5 - 5)
            val = np.clip(val, 0.1, 1.0)
            vals.append(val)
        else:
            vals.append(0.0)
    return vals