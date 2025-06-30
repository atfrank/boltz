import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif, to_mmcif_trajectory
from boltz.data.write.pdb import to_pdb


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]
        
        # Handle SAXS logs if available in predictions
        if "saxs_logs" in prediction:
            # Save SAXS logs for all samples
            for record in records:
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)
                
                saxs_logs = prediction["saxs_logs"]
                if saxs_logs:
                    for sample_id, log_data in saxs_logs.items():
                        log_file = struct_dir / f"saxs_chi2_sample_{sample_id}.json"
                        with log_file.open('w') as f:
                            json.dump(log_data, f, indent=2)
                        print(f"SAXS chi-squared log saved to {log_file}")
                break  # Only need to save once for all records

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                
                # Save trajectory if available for this specific sample
                if "trajectory_coords" in prediction:
                    trajectory_coords_tensor = prediction["trajectory_coords"]
                    if hasattr(trajectory_coords_tensor, 'cpu'):
                        trajectory_coords_all = trajectory_coords_tensor.cpu().numpy()
                    else:
                        trajectory_coords_all = trajectory_coords_tensor
                    
                    # Extract trajectory for current sample (model_idx)
                    # trajectory shape: [n_steps, n_samples, n_atoms, 3]
                    if len(trajectory_coords_all.shape) == 4:
                        # Multiple samples
                        if model_idx < trajectory_coords_all.shape[1]:
                            trajectory_coords = trajectory_coords_all[:, model_idx]  # [n_steps, n_atoms, 3]
                        else:
                            trajectory_coords = None
                    else:
                        # Single sample case (backward compatibility)
                        if model_idx == 0:
                            trajectory_coords = trajectory_coords_all  # [n_steps, n_atoms, 3]
                        else:
                            trajectory_coords = None
                    
                    if trajectory_coords is not None:
                        # Create trajectory structures for each frame
                        trajectory_structures = []
                        for frame_idx in range(trajectory_coords.shape[0]):
                            frame_coords = trajectory_coords[frame_idx]
                            
                            # Unpad coordinates (ensure pad_mask is on CPU)
                            pad_mask_cpu = pad_mask.cpu() if hasattr(pad_mask, 'cpu') else pad_mask
                            frame_coords_unpad = frame_coords[pad_mask_cpu.bool()]
                            
                            # Create structure for this frame
                            frame_atoms = structure.atoms.copy()
                            frame_atoms["coords"] = frame_coords_unpad
                            frame_atoms["is_present"] = True
                            
                            if self.boltz2:
                                frame_coords_typed = [(x,) for x in frame_coords_unpad]
                                frame_coords_typed = np.array(frame_coords_typed, dtype=Coords)
                                frame_structure = replace(
                                    structure,
                                    atoms=frame_atoms,
                                    residues=residues,
                                    interfaces=interfaces,
                                    coords=frame_coords_typed,
                                )
                            else:
                                frame_structure = replace(
                                    structure,
                                    atoms=frame_atoms,
                                    residues=residues,
                                    interfaces=interfaces,
                                )
                            trajectory_structures.append(frame_structure)
                    
                        # Write trajectory file with sample number
                        if self.output_format == "pdb":
                            traj_path = struct_dir / f"{record.id}_model_{model_idx}_trajectory.pdb"
                            with traj_path.open("w") as f:
                                for frame_idx, frame_structure in enumerate(trajectory_structures):
                                    f.write(f"MODEL {frame_idx + 1}\n")
                                    pdb_content = to_pdb(frame_structure, boltz2=self.boltz2)
                                    # Remove the final END line from individual models
                                    pdb_lines = pdb_content.strip().split('\n')
                                    if pdb_lines and pdb_lines[-1] == "END":
                                        pdb_lines = pdb_lines[:-1]
                                    f.write('\n'.join(pdb_lines) + '\n')
                                    f.write("ENDMDL\n")
                                f.write("END\n")
                        elif self.output_format == "mmcif":
                            traj_path = struct_dir / f"{record.id}_model_{model_idx}_trajectory.cif"
                            with traj_path.open("w") as f:
                                # For now, write each model as separate mmCIF entries
                                # Full multi-model mmCIF requires complex ModelCIF library coordination
                                for frame_idx, frame_structure in enumerate(trajectory_structures):
                                    if frame_idx == 0:
                                        # Write the first model with full mmCIF structure
                                        f.write(to_mmcif(frame_structure, boltz2=self.boltz2))
                                    else:
                                        # For subsequent models, would need to append coordinate data
                                        # with proper model numbering - this is complex with ModelCIF
                                        pass
                                # Note: Full multi-model implementation would require coordination
                                # of all mmCIF data sections with proper model numbering

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    # Add Rg guidance information if available
                    print(f"Writer: Checking prediction keys: {list(prediction.keys())}")
                    if "rg_guidance" in prediction:
                        rg_info = prediction["rg_guidance"]
                        print(f"Writer: Found rg_guidance: {rg_info}")
                        confidence_summary_dict["rg_guidance"] = {
                            "final_rg": rg_info["final_rg"],
                            "target_rg": rg_info["target_rg"],
                            "rg_error": rg_info["rg_error"],
                            "rg_relative_error": rg_info["rg_relative_error"]
                        }
                    else:
                        print("Writer: No rg_guidance found in prediction")
                    
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
