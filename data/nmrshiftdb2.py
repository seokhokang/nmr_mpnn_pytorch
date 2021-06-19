import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
    
# the dataset can be downloaded from
# https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help


def get_atom_shifts(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}
    for key in molprops.keys():
    
        if key.startswith('Spectrum 13C'):
            
            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
            
                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                atom_shifts[shift_idx].append(shift_val)

    return atom_shifts


def add_mol(mol_dict, mol):

    def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

    def _RS(atom):
        
        if atom.HasProp('_CIPCode'): RS_list = [(atom.GetProp('_CIPCode') == 'R'), (atom.GetProp('_CIPCode') == 'S')] 
        else: RS_list = [0, 0]

        return RS_list

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    atom_fea1 = np.eye(len(atom_list), dtype=np.int)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(hybridization_list), dtype=np.int)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]]
    atom_fea3 = [[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()] 
    atom_fea4 = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
                  a.GetTotalNumHs(), a.GetImplicitValence(), a.GetNumRadicalElectrons(),
                  a.IsInRing(), a.GetIsAromatic()] for a in mol.GetAtoms()]   
    D_list, A_list = _DA(mol)    
    atom_fea5 = [[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())]
    atom_fea6 = [_RS(a) for a in mol.GetAtoms()]
    
    node_attr = np.concatenate([atom_fea1, atom_fea2[:,2:], atom_fea3, atom_fea4, atom_fea5, atom_fea6], 1)

    shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    
    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype=np.int)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.eye(len(stereo_list), dtype=np.int)[[stereo_list.index(str(b.GetStereo())) for b in mol.GetBonds()]]
        bond_fea3 = [[b.GetIsConjugated(), b.IsInRing()] for b in mol.GetBonds()]   
        
        edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=np.int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict


molsuppl = Chem.SDMolSupplier('nmrshiftdb2withsignals.sd', removeHs = False)

atom_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi']
hybridization_list = ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2']
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
stereo_list = ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE']

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

mol_dict = {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': [],
            'shift': [],
            'mask': [],
            'smi': []}
                 
for i, mol in enumerate(molsuppl):

    try:
        Chem.SanitizeMol(mol)
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        Chem.rdmolops.AssignStereochemistry(mol)
        assert '.' not in Chem.MolToSmiles(mol)
    except:
        continue

    atom_shifts = get_atom_shifts(mol)
    if len(atom_shifts) == 0: continue
    for j, atom in enumerate(mol.GetAtoms()):
        if j in atom_shifts:
            atom.SetDoubleProp('shift', np.median(atom_shifts[j]))
            atom.SetBoolProp('mask', 1)
        else:
            atom.SetDoubleProp('shift', 0)
            atom.SetBoolProp('mask', 0)

    mol = Chem.RemoveHs(mol)
    mol_dict = add_mol(mol_dict, mol)

    if (i+1) % 1000 == 0: print('%d/%d processed' %(i+1, len(molsuppl)))

print('%d/%d processed' %(i+1, len(molsuppl)))   

mol_dict['n_node'] = np.array(mol_dict['n_node'])
mol_dict['n_edge'] = np.array(mol_dict['n_edge'])
mol_dict['node_attr'] = np.vstack(mol_dict['node_attr'])
mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr'])
mol_dict['src'] = np.hstack(mol_dict['src'])
mol_dict['dst'] = np.hstack(mol_dict['dst'])
mol_dict['shift'] = np.hstack(mol_dict['shift'])
mol_dict['mask'] = np.hstack(mol_dict['mask'])
mol_dict['smi'] = np.array(mol_dict['smi'])

for key in mol_dict.keys():  print(key, mol_dict[key].shape, mol_dict[key].dtype)
    
np.savez_compressed('./dataset_graph.npz', data = [mol_dict])