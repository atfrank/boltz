# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        init_coords_path=None,
        noise_specs=None,
        structure=None,
        guidance=None,
        save_trajectory=False,
        non_target_noise_min=0.1,
        non_target_noise_range=0.2,
        start_sigma_scale=1.0,
        no_random_augmentation=False,
        residue_based_selection=False,
        **network_condition_kwargs,
    ):
        # Set default steering args if not provided
        if steering_args is None:
            steering_args = {
                "fk_steering": False,
                "guidance_update": True,  # Enable guidance by default when SAXS is present
                "num_gd_steps": 1,
                "num_particles": 1,
                "fk_resampling_interval": 1,
                "fk_lambda": 1.0
            }
        
        # Enable guidance if SAXS or Rg is present (override default steering args)
        if guidance is not None:
            guidance_obj = guidance[0] if isinstance(guidance, list) else guidance
            has_saxs = guidance_obj is not None and hasattr(guidance_obj, 'saxs') and guidance_obj.saxs is not None
            has_rg = guidance_obj is not None and hasattr(guidance_obj, 'rg') and guidance_obj.rg is not None
            
            if has_saxs or has_rg:
                # Override steering args to enable guidance
                steering_args = dict(steering_args)  # Make a copy
                steering_args["guidance_update"] = True
                if steering_args["num_gd_steps"] < 1:
                    steering_args["num_gd_steps"] = 1
        
        # Extract guidance configs if available
        saxs_guidance_config = None
        rg_guidance_config = None
        if guidance is not None:
            # Handle case where guidance comes as a list (from batch collation)
            if isinstance(guidance, list):
                guidance_obj = guidance[0]  # Take first item from batch
            else:
                guidance_obj = guidance
                
            if guidance_obj is not None:
                if hasattr(guidance_obj, 'saxs') and guidance_obj.saxs is not None:
                    saxs_guidance_config = guidance_obj.saxs
                    print(f"SAXS guidance activated: weight={saxs_guidance_config.guidance_weight}, interval={saxs_guidance_config.guidance_interval}")
                
                if hasattr(guidance_obj, 'rg') and guidance_obj.rg is not None:
                    rg_guidance_config = guidance_obj.rg
                    print(f"Rg guidance activated: target={rg_guidance_config.target_rg}, force_constant={rg_guidance_config.force_constant}")
                else:
                    print(f"No Rg guidance found in guidance object. Has rg attribute: {hasattr(guidance_obj, 'rg')}")
        
        potentials = get_potentials(saxs_guidance_config=saxs_guidance_config, rg_guidance_config=rg_guidance_config)
        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if steering_args["guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        
        # Scale entire sigma schedule if using init_coords
        if init_coords_path is not None and start_sigma_scale != 1.0:
            sigmas = sigmas * start_sigma_scale
            
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # Initialize atom coordinates
        init_sigma = sigmas[0]
        
        if init_coords_path is not None:
            # Import needed for PDB/mmCIF parsing
            from boltz.data.write.utils import read_coords_from_file
            
            try:
                # Read the coordinates from the PDB/mmCIF file
                expected_atoms = shape[1]
                init_coords = read_coords_from_file(init_coords_path, expected_atoms)
                
                if init_coords is not None:
                    # Create tensor and expand for batch
                    init_coords_tensor = torch.tensor(init_coords, dtype=torch.float32)
                    if torch.cuda.is_available() and next(self.parameters()).is_cuda:
                        init_coords_tensor = init_coords_tensor.to(self.device)
                    
                    # Reshape to match expected dimensions
                    init_coords_tensor = init_coords_tensor.unsqueeze(0)
                    init_coords_tensor = init_coords_tensor.expand(shape[0], -1, -1)
                    
                    # Use initial coordinates without adding global noise
                    atom_coords = init_coords_tensor
                    print(f"Using initial coordinates from {init_coords_path}")
                else:
                    print(f"Warning: Failed to load coordinates from {init_coords_path}")
                    print("Using random initialization instead.")
                    atom_coords = init_sigma * torch.randn(shape, device=self.device)
            except Exception as e:
                print(f"Error loading initial coordinates: {e}. Using random initialization instead.")
                atom_coords = init_sigma * torch.randn(shape, device=self.device)
        else:
            # Original code for random initialization
            atom_coords = init_sigma * torch.randn(shape, device=self.device)
        
        # Setup selective denoising variables
        use_selective_denoising = init_coords_path is not None
        targeted_atoms_mask = None
        
        # Show selective refinement info even if structure is None
        if init_coords_path is not None or noise_specs:
            print(f"")
            print(f"=== SELECTIVE REFINEMENT PARAMETERS ===")
            print(f"Init coordinates: {init_coords_path}")
            print(f"Noise specifications: {noise_specs}")
            print(f"Start sigma scale: {start_sigma_scale}")
            print(f"Random augmentation: {'disabled' if no_random_augmentation else 'enabled'}")
            print(f"Residue-based selection: {'enabled' if residue_based_selection else 'disabled'}")
            print(f"Non-target noise min: {non_target_noise_min}")
            print(f"Non-target noise range: {non_target_noise_range}")
            print(f"Structure available: {'yes' if structure is not None else 'no'}")
            print(f"========================================")
            print(f"")
        
        # Apply selective noise if specified
        if noise_specs:
            try:
                from boltz.data.noise import parse_noise_specifications, apply_noise_to_selection
                
                # Parse noise specifications
                parsed_noise_specs = parse_noise_specifications(noise_specs)
                print(f"Parsed {len(parsed_noise_specs)} noise specifications")
                
                if structure is not None:
                    # Handle structure being a list (batch dimension)
                    if isinstance(structure, list) and len(structure) > 0:
                        structure_obj = structure[0]  # Use first structure in batch
                    else:
                        structure_obj = structure
                    
                    if structure_obj is not None:
                        # First apply global centering to maintain coherence
                        atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
                        
                        # Then apply targeted noise
                        atom_coords, targeted_atoms_mask = apply_noise_to_selection(
                            atom_coords, parsed_noise_specs, structure_obj, self.device,
                            residue_based=residue_based_selection
                        )
                        total_atoms = targeted_atoms_mask.shape[1]
                        targeted_count = targeted_atoms_mask.sum().item()
                        non_targeted_count = total_atoms - targeted_count
                        percentage_targeted = (targeted_count / total_atoms) * 100 if total_atoms > 0 else 0
                        
                        print(f"")
                        print(f"=== SELECTIVE REFINEMENT SUMMARY ===")
                        print(f"Total atoms: {total_atoms}")
                        print(f"Targeted atoms: {targeted_count} ({percentage_targeted:.1f}%)")
                        print(f"Non-targeted atoms: {non_targeted_count} ({100-percentage_targeted:.1f}%)")
                        print(f"Residue-based selection: {'enabled' if residue_based_selection else 'disabled'}")
                        print(f"Random augmentation: {'disabled' if no_random_augmentation else 'enabled'}")
                        print(f"Noise specifications: {len(parsed_noise_specs)}")
                        print(f"=====================================")
                        print(f"")
                        
                        # Expand mask to match coordinate batch size (multiplicity)
                        if targeted_atoms_mask.shape[0] != atom_coords.shape[0]:
                            targeted_atoms_mask = targeted_atoms_mask.expand(atom_coords.shape[0], -1)
                    else:
                        print("Warning: Structure object is None, cannot apply selective noise")
                else:
                    print("Warning: Structure not available in feats, cannot apply selective noise")
                    print("Note: Atom selection information will not be shown")
                    # At least show what we tried to parse
                    for i, spec in enumerate(parsed_noise_specs):
                        print(f"  Specification {i+1}: {spec.selection_type} - {spec}")
            except Exception as e:
                print(f"Warning: Failed to apply selective noise: {e}")
                import traceback
                traceback.print_exc()
        token_repr = None
        atom_coords_denoised = None
        
        # Initialize trajectory storage for all samples
        trajectory_coords = []
        trajectory_denoised_coords = []
        if save_trajectory:
            # Store initial coordinates (centered, only first sample to ensure consistency)
            centered_initial = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            # Only store first sample to avoid dimension issues with resampling
            trajectory_coords.append(centered_initial[:1].clone().cpu())
            # No denoised coordinates for initial step (pure noise)
            trajectory_denoised_coords.append(None)

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            # Skip random augmentation if requested
            if no_random_augmentation and init_coords_path is not None:
                # Use identity rotation and zero translation
                random_R = torch.eye(3, device=atom_coords.device, dtype=atom_coords.dtype).unsqueeze(0).expand(multiplicity, -1, -1)
                random_tr = torch.zeros((multiplicity, 1, 3), device=atom_coords.device, dtype=atom_coords.dtype)
            else:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if steering_args["guidance_update"] and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            
            # Apply differential noise scaling if using targeted denoising
            if use_selective_denoising and targeted_atoms_mask is not None:
                # Create noise scaling mask
                noise_scale = torch.ones_like(targeted_atoms_mask, dtype=torch.float32)
                
                # Targeted atoms get full noise (scale = 1.0)
                # Non-targeted atoms get reduced noise based on provided parameters
                step_progress = step_idx / num_sampling_steps
                non_target_scale = non_target_noise_min + non_target_noise_range * step_progress
                noise_scale[~targeted_atoms_mask] = non_target_scale
                
                # Log differential noise scaling every 10 steps or at key points
                if step_idx == 0 or step_idx % 10 == 0 or step_idx == num_sampling_steps - 1:
                    targeted_count = targeted_atoms_mask.sum().item()
                    non_targeted_count = (~targeted_atoms_mask).sum().item()
                    print(f"Step {step_idx+1}/{num_sampling_steps}: Noise scaling - Targeted: 1.0 ({targeted_count} atoms), Non-targeted: {non_target_scale:.3f} ({non_targeted_count} atoms)")
                
                # Expand to match coordinate dimensions and apply scaling
                noise_scale_expanded = noise_scale.unsqueeze(-1).expand_as(eps)
                eps = eps * noise_scale_expanded
            
            # Apply noise to entire system (maintains physical coherence)
            atom_coords_noisy = atom_coords + eps

            # Apply raw coordinate guidance before neural network denoising
            for potential in potentials:
                if hasattr(potential, 'apply_raw_guidance'):
                    parameters = potential.compute_parameters(steering_t)
                    if parameters and parameters.get("raw_guidance_weight", 0) > 0:
                        atom_coords_noisy = potential.apply_raw_guidance(
                            atom_coords_noisy, 
                            network_condition_kwargs["feats"],
                            parameters,
                            sigma_t,  # Current noise level
                            step_idx
                        )

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0 or energy_traj.shape[1] < 2:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if steering_args["guidance_update"] and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["guidance_update"]
                    and step_idx < num_sampling_steps - 1
                ):
                    # Log Rg on raw denoised coordinates before gradient descent optimization
                    for potential in potentials:
                        if hasattr(potential, 'log_raw_rg'):
                            potential.log_raw_rg(step_idx, atom_coords_denoised, network_condition_kwargs["feats"])
                    
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            # Set step information for potentials with step tracking (SAXS, Rg, etc.)
                            if hasattr(potential, 'set_step_info'):
                                potential.set_step_info(step_idx, guidance_step)
                            
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if steering_args["guidance_update"]:
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next
            
            # Update potentials with current step info after diffusion step is complete
            for potential in potentials:
                if hasattr(potential, 'set_step_info'):
                    # Set final step info after diffusion step completion
                    potential.set_step_info(step_idx + 1, 0)  # Next diffusion step, guidance step 0
            
            # Save trajectory step if requested (after centering like final coordinates)
            if save_trajectory:
                # Center coordinates like final output and store all samples
                centered_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
                centered_denoised = atom_coords_denoised - atom_coords_denoised.mean(dim=-2, keepdims=True)
                
                # Ensure consistent batch size by only saving the first sample if resampling changed sizes
                if len(trajectory_coords) > 0 and centered_coords.shape[0] != trajectory_coords[0].shape[0]:
                    # Take only the first sample to maintain consistent dimensions
                    centered_coords = centered_coords[:1]
                    centered_denoised = centered_denoised[:1]
                    
                trajectory_coords.append(centered_coords.clone().cpu())
                trajectory_denoised_coords.append(centered_denoised.clone().cpu())

        # Prepare output dictionary
        output_dict = dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)
        
        # Add trajectory if saved
        if save_trajectory and trajectory_coords:
            # Stack trajectory frames: trajectory_coords is list of [num_samples, num_atoms, 3]
            # We want final shape [num_timesteps, num_samples, num_atoms, 3]
            trajectory_tensor = torch.stack(trajectory_coords, dim=0)
            output_dict['trajectory_coords'] = trajectory_tensor
            
            # Stack denoised trajectory (skip None entries)
            denoised_frames = [frame for frame in trajectory_denoised_coords if frame is not None]
            if denoised_frames:
                trajectory_denoised_tensor = torch.stack(denoised_frames, dim=0)
                output_dict['trajectory_denoised_coords'] = trajectory_denoised_tensor
                print(f"Saved raw trajectory with {len(trajectory_coords)} steps and denoised trajectory with {len(denoised_frames)} steps for {multiplicity} samples")
            else:
                print(f"Saved raw trajectory with {len(trajectory_coords)} steps for {multiplicity} samples")
        
        # Collect SAXS chi-squared logs if available
        saxs_logs = {}
        for potential in potentials:
            if hasattr(potential, '_chi2_logs') and potential._chi2_logs:
                saxs_logs.update(potential._chi2_logs)
                
                # Save chi2 logs to file
                if hasattr(potential, 'save_chi2_log'):
                    potential.save_chi2_log()
                
                # Print summary
                summary = potential.get_chi2_summary()
                if summary:
                    for sample_id, stats in summary.items():
                        print(f"SAXS guidance sample {sample_id}: min_chi2={stats['min_chi2']:.2f}, "
                              f"final_chi2={stats['final_chi2']:.2f}, "
                              f"evaluations={stats['num_evaluations']}")
        
        # Add SAXS logs to output if available
        if saxs_logs:
            output_dict['saxs_logs'] = saxs_logs
        
        # Collect Rg guidance information if available
        rg_info = {}
        print(f"Diffusion: Checking {len(potentials)} potentials for Rg guidance...")
        for i, potential in enumerate(potentials):
            print(f"  Potential {i}: {type(potential).__name__}, "
                  f"has_get_final_rg={hasattr(potential, 'get_final_rg')}, "
                  f"has_get_target_rg={hasattr(potential, 'get_target_rg')}")
            if hasattr(potential, 'get_final_rg') and hasattr(potential, 'get_target_rg'):
                final_rg = potential.get_final_rg()
                target_rg = potential.get_target_rg()
                print(f"  Potential {i}: final_rg={final_rg}, target_rg={target_rg}")
                if final_rg is not None:
                    rg_info = {
                        'final_rg': final_rg,
                        'target_rg': target_rg,
                        'rg_error': abs(final_rg - target_rg) if target_rg else None,
                        'rg_relative_error': abs(final_rg - target_rg) / target_rg * 100 if target_rg else None
                    }
                    print(f"Rg guidance final result: Rg={final_rg:.2f}Å, "
                          f"target={target_rg:.2f}Å, "
                          f"error={abs(final_rg - target_rg):.2f}Å "
                          f"({abs(final_rg - target_rg) / target_rg * 100:.1f}%)")
                    break
        
        # Add Rg info to output if available
        if rg_info:
            output_dict['rg_guidance'] = rg_info
            print(f"Diffusion: Added rg_guidance to output_dict: {rg_info}")
        else:
            print("Diffusion: No rg_info to add to output_dict")
        
        print(f"Diffusion: Final output_dict keys: {list(output_dict.keys())}")
        return output_dict

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords, _ = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            noised_atom_coords = out_dict["noised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
