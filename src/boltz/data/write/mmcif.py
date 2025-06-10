import io
import re
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure


def to_mmcif_trajectory(
    structures: list[Structure],
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write multiple structures into a multi-model MMCIF file.

    Parameters
    ----------
    structures : list[Structure]
        The input structures (trajectory frames)
    plddts : Optional[Tensor]
        The pLDDT values for each atom (optional)
    boltz2 : bool
        Whether this is a Boltz2 structure

    Returns
    -------
    str
        the output multi-model MMCIF file

    """
    if not structures:
        raise ValueError("At least one structure is required")
    
    # Use the first structure as template for entities and chains
    template_structure = structures[0]
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in template_structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences (using residue names, not letters)
    entity_to_sequence = {}
    for entity_id, chains in entity_to_chains.items():
        # Use the first chain to determine sequence
        chain = chains[0]
        mol_type = entity_to_moltype[entity_id]

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = template_structure.residues[res_start:res_end]

        # Use residue names directly (like "ALA", "CYS", "U", "C", etc)
        sequence = [str(res["name"]) for res in residues]
        entity_to_sequence[entity_id] = sequence

    # Map entity to entity
    entity_map = {}

    # Create entities using the same approach as the original function
    lig_entity = None
    entities = []
    for entity_id, sequence in entity_to_sequence.items():
        mol_type = entity_to_moltype[entity_id]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        # Handle smiles
        if len(sequence) == 1 and (sequence[0] == "LIG"):
            if lig_entity is None:
                seq = [chem_comp(sequence[0])]
                lig_entity = Entity(seq)
            entity = lig_entity
        else:
            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            entity = Entity(seq)

        entity_map[entity_id] = entity
        entities.append(entity)

    system.entities.extend(entities)

    # Create asymmetric units for each chain
    asym_units = {}
    chain_details = {}
    for entity_id, chains in entity_to_chains.items():
        entity = entity_map[entity_id]
        for chain in chains:
            chain_id = chain["asym_id"]
            asym_unit = AsymUnit(entity=entity, details=f"Chain {chain['name']}")
            asym_units[chain_id] = asym_unit
            chain_details[chain_id] = chain

    system.asym_units.extend(asym_units.values())

    # Create assembly
    assembly = Assembly(asym_units.values(), name="Complete assembly")
    system.assemblies.append(assembly)

    # Create models for each structure in the trajectory
    models = []
    for model_idx, structure in enumerate(structures):
        model = AbInitioModel(assembly=assembly, name=f"Model {model_idx + 1}")

        # Add atoms for this model
        for chain_id, asym_unit in asym_units.items():
            chain = chain_details[chain_id]
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            residues = structure.residues[res_start:res_end]

            for res_idx, residue in enumerate(residues):
                atom_start = residue["atom_idx"]
                atom_end = residue["atom_idx"] + residue["atom_num"]
                atoms = structure.atoms[atom_start:atom_end]
                atom_coords = atoms["coords"]

                for atom_idx, atom in enumerate(atoms):
                    if not atom["is_present"]:
                        continue

                    if boltz2:
                        atom_name = str(atom["name"])
                        atom_key = re.sub(r"\d", "", atom_name)
                        if atom_key in const.ambiguous_atoms:
                            if isinstance(const.ambiguous_atoms[atom_key], str):
                                element = const.ambiguous_atoms[atom_key]
                            elif str(residue["name"]) in const.ambiguous_atoms[atom_key]:
                                element = const.ambiguous_atoms[atom_key][str(residue["name"])]
                            else:
                                element = const.ambiguous_atoms[atom_key]["*"]
                        else:
                            element = atom_key[0]
                    else:
                        atom_name = [chr(c + 32) for c in atom["name"] if c != 0]
                        atom_name = "".join(atom_name)
                        element = periodic_table.GetElementSymbol(atom["element"].item())

                    # Get coordinates - handle different coordinate structures
                    coords = atom_coords[atom_idx]
                    if boltz2:
                        # Handle Boltz2 coordinate structure which may have extra dimensions
                        while hasattr(coords, '__len__') and hasattr(coords, 'shape') and len(coords.shape) > 1 and coords.shape[-1] != 3:
                            coords = coords[0]
                        if hasattr(coords, '__len__') and hasattr(coords, 'shape') and coords.shape[-1] == 3:
                            # Ensure we have a 3D coordinate
                            if len(coords.shape) > 1:
                                coords = coords.flatten()[:3]

                    # Get B-factor from pLDDT if available
                    b_factor = 100.0
                    if plddts is not None:
                        # For multi-model, we'd need plddt per model - for now use the same
                        residue_plddt_idx = res_start + res_idx
                        if residue_plddt_idx < len(plddts):
                            b_factor = round(plddts[residue_plddt_idx].item() * 100, 2)

                    # Create atom with model number
                    model_atom = Atom(
                        asym_unit=asym_unit,
                        seq_id=res_idx + 1,
                        atom_id=atom_name.strip(),
                        type_symbol=element,
                        x=float(coords[0]),
                        y=float(coords[1]),
                        z=float(coords[2]),
                        biso=b_factor,
                        occupancy=1.0,
                    )
                    # Set the model number for this atom
                    model_atom.pdbx_PDB_model_num = model_idx + 1
                    model.add_atom(model_atom)

        models.append(model)

    # Create model group with all models
    model_group = ModelGroup(models, name="Trajectory models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()


def to_mmcif(
    structure: Structure,
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output MMCIF file

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Create entity objects
    lig_entity = None
    entities_map = {}
    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        # Handle smiles
        if len(sequence) == 1 and (sequence[0] == "LIG"):
            if lig_entity is None:
                seq = [chem_comp(sequence[0])]
                lig_entity = Entity(seq)
            model_e = lig_entity
        else:
            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            model_e = Entity(seq)

        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        chain_tag = str(chain["name"])
        entity = entities_map[chain_idx]
        if entity.type == "water":
            asym = ihm.WaterAsymUnit(
                entity,
                1,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        else:
            asym = AsymUnit(
                entity,
                details="Model subunit %s" % chain_tag,
                id=chain_tag,
            )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Index into plddt tensor for current residue.
            res_num = 0
            # Tracks non-ligand plddt tensor indices,
            # Initializing to -1 handles case where ligand is resnum 0
            prev_polymer_resnum = -1
            # Tracks ligand indices.
            ligand_index_offset = 0

            # Add all atom sites.
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    res_name = str(residue["name"])
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        if boltz2:
                            atom_name = str(atom["name"])
                            atom_key = re.sub(r"\d", "", atom_name)
                            if atom_key in const.ambiguous_atoms:
                                if isinstance(const.ambiguous_atoms[atom_key], str):
                                    element = const.ambiguous_atoms[atom_key]
                                elif res_name in const.ambiguous_atoms[atom_key]:
                                    element = const.ambiguous_atoms[atom_key][res_name]
                                else:
                                    element = const.ambiguous_atoms[atom_key]["*"]
                            else:
                                element = atom_key[0]
                        else:
                            atom_name = atom["name"]
                            atom_name = [chr(c + 32) for c in atom_name if c != 0]
                            atom_name = "".join(atom_name)
                            element = periodic_table.GetElementSymbol(
                                atom["element"].item()
                            )
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]

                        if record_type != "HETATM":
                            # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                            biso = (
                                100.00
                                if plddts is None
                                else round(
                                    plddts[res_num + ligand_index_offset].item() * 100,
                                    3,
                                )
                            )
                            prev_polymer_resnum = res_num
                        else:
                            # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                            ligand_index_offset += 1
                            biso = (
                                100.00
                                if plddts is None
                                else round(
                                    plddts[
                                        prev_polymer_resnum + ligand_index_offset
                                    ].item()
                                    * 100,
                                    3,
                                )
                            )

                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=atom_name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    if record_type != "HETATM":
                        res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            prev_polymer_resnum = (
                -1
            )  # -1 handles case where ligand is the first residue
            ligand_index_offset = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )

                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1

                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]

                    if record_type != "HETATM":
                        # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[res_num + ligand_index_offset].item() * 100,
                                    3,
                                ),
                            )
                        )
                        prev_polymer_resnum = res_num
                    else:
                        # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(residue_idx),
                                round(
                                    plddts[
                                        prev_polymer_resnum
                                        + ligand_index_offset
                                        + 1 : prev_polymer_resnum
                                        + ligand_index_offset
                                        + residue["atom_num"]
                                        + 1
                                    ]
                                    .mean()
                                    .item()
                                    * 100,
                                    2,
                                ),
                            )
                        )
                        ligand_index_offset += residue["atom_num"]

                    if record_type != "HETATM":
                        res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()
