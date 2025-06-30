# Rg Potential Analysis and PDB Target Integration

## Analysis Summary

### ✅ **Rg Potential Integration is Fully Consistent with Boltz Framework**

After comprehensive analysis of all potentials in the Boltz codebase, the Rg potential implementations follow exactly the same patterns as other potentials:

#### **Framework Consistency Check**

| Feature | Built-in Potentials | SAXS | Rg | ✅ Status |
|---------|-------------------|------|----|-----------| 
| Framework inheritance | ✅ | ✅ | ✅ | **Consistent** |
| Dataclass config | ✅ | ✅ | ✅ | **Consistent** |
| get_potentials() integration | ✅ | ✅ | ✅ | **Consistent** |
| Parameter scheduling | ✅ | ✅ | ✅ | **Consistent** |
| Wrapper pattern | ✅ | ✅ | ✅ | **Consistent** |
| Gradient computation | ✅ | ✅ | ✅ | **Consistent** |

#### **Integration Patterns**

1. **Configuration**: Uses `RgGuidanceConfig` dataclass like SAXS uses `SAXSGuidanceConfig`
2. **Framework Inheritance**: Both use the `Potential` abstract base class
3. **Integration Method**: Both added to `get_potentials()` function with identical parameter patterns
4. **Parameter Structure**: Both use standard framework parameters:
   - `guidance_interval`: How often to apply (1 = every step)
   - `guidance_weight`: Weight scaling factor  
   - `resampling_weight`: Weight during resampling steps
5. **Wrapper Pattern**: Both use wrapper classes for framework integration

#### **Advanced Rg Features**

The Rg potential implementations include additional robustness features beyond standard potentials:

- **Multiple Implementations**: Basic (`RgPotentialWrapper`) and robust (`RobustRgPotentialWrapper`) versions
- **Outlier Protection**: Robust version prevents "extreme atom cheating"
- **Force Ramping**: Gradual increase of force constants for stability
- **Gradient Capping**: Prevents excessive forces that could destabilize diffusion
- **Displacement Monitoring**: Tracks and limits per-step atom displacements

### **Atom Selection for Rg Calculations**

✅ **Confirmed**: Both Rg potential implementations correctly use **all atoms** by default:

- **Base RadiusOfGyrationPotential**: `atom_selection: Optional[str] = None` (defaults to all atoms)
- **RobustRgPotential**: `atom_selection: str = "all"` (explicitly all atoms)

This is the correct behavior for Rg guidance, as it should:
1. **Calculate Rg using all atoms** for proper structural representation
2. **Apply forces to all atoms** to achieve target Rg through global structural changes

## PDB Target Integration

### **New Feature: Calculate Target Rg from Reference PDB Structures**

Added comprehensive PDB integration that allows users to specify a reference PDB structure instead of manually providing target Rg values.

#### **New Configuration Parameters**

Added to `RgGuidanceConfig`:

```python
# PDB target calculation
reference_pdb_path: Optional[str] = None  # Path to reference PDB structure
pdb_chain_id: Optional[str] = None  # Specific chain to use for Rg calculation
pdb_atom_selection: Optional[str] = None  # Atom selection for PDB ('all', 'backbone', 'heavy')
```

#### **Implementation Components**

1. **PDB Utilities** (`pdb_utils.py`):
   - `parse_pdb_coordinates()`: Extract coordinates from PDB files
   - `calculate_radius_of_gyration()`: Compute Rg with mass weighting
   - `calculate_target_rg_from_pdb()`: Complete PDB→Rg pipeline
   - `get_default_atom_masses()`: Standard atomic masses
   - `validate_pdb_file()`, `list_chains_in_pdb()`: Helper functions

2. **Atom Selection Support**:
   - **"all"**: All atoms (recommended for Rg guidance)
   - **"backbone"**: Protein (N, CA, C, O) and nucleic acid (P, O5', C5', C4', C3', O3', etc.) backbone atoms
   - **"heavy"**: All non-hydrogen atoms

3. **Integration Points**:
   - **YAML Parsing** (`schema.py`): Parse new PDB parameters
   - **JSON Serialization** (`main.py`): Serialize PDB parameters
   - **JSON Deserialization** (`inferencev2.py`): Deserialize PDB parameters
   - **Potential Wrappers**: Calculate target from PDB during initialization

#### **Usage Examples**

**YAML Configuration:**
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKF

guidance:
  rg:
    # Calculate target from reference PDB
    reference_pdb_path: "./reference_structure.pdb"
    pdb_chain_id: "A"  # Optional: specific chain
    pdb_atom_selection: "all"  # Recommended: use all atoms
    
    # Force parameters
    force_constant: 100.0
    mass_weighted: true
    
    # Robustness parameters
    robust_mode: true
    force_ramping: false
    max_displacement_per_step: 3.0
    gradient_capping: 20.0
```

**Alternative: Manual Target:**
```yaml
guidance:
  rg:
    target_rg: 25.0  # Manual target in Angstroms
    force_constant: 100.0
    # ... other parameters
```

**Alternative: SAXS-derived Target:**
```yaml
guidance:
  rg:
    saxs_file_path: "./experimental_saxs.dat"  # Extract Rg via Guinier analysis
    force_constant: 100.0
    # ... other parameters
```

#### **Error Handling and Fallbacks**

The implementation includes robust error handling:

1. **PDB Calculation Failure**: Falls back to manual `target_rg` if provided
2. **File Not Found**: Clear error message with fallback options
3. **Invalid Selections**: Validates atom selections and provides helpful errors
4. **Chain Validation**: Lists available chains if specified chain not found

#### **Validation and Testing**

Created comprehensive test suite (`test_pdb_rg_calculation.py`) that validates:

1. **PDB Parsing**: Coordinate extraction, chain detection, atom counting
2. **Rg Calculation**: All atom selections, mass weighting, different structures
3. **YAML Integration**: Dataclass creation with PDB parameters
4. **Wrapper Integration**: End-to-end PDB→Rg→Potential pipeline

**Test Results:**
```
Testing all atoms:
  Target Rg: 4.77 Å (46/46 atoms, mass-weighted)

Testing backbone atoms:
  Target Rg: 3.91 Å (18/46 atoms, nucleic acid backbone)

Testing heavy atoms:
  Target Rg: 4.77 Å (46/46 atoms, no hydrogens in test PDB)

Wrapper Integration:
  ✅ Successfully created RobustRgPotentialWrapper with PDB target
  ✅ Target automatically calculated: 4.77 Å
```

## Benefits of PDB Integration

1. **User Convenience**: No need to manually calculate or estimate target Rg values
2. **Accuracy**: Uses actual experimental/computational structure data
3. **Consistency**: Same mass weighting and calculation methods as the guidance potential
4. **Flexibility**: Supports different atom selections for specialized use cases
5. **Robustness**: Comprehensive error handling and fallback mechanisms
6. **Documentation**: Clear examples and usage patterns

## Recommendations

1. **Default Usage**: Use `pdb_atom_selection: "all"` for standard Rg guidance
2. **Mass Weighting**: Keep `mass_weighted: true` for physically accurate calculations
3. **Error Handling**: Always provide fallback `target_rg` when using PDB targets
4. **File Paths**: Use relative paths from the working directory for portability
5. **Chain Selection**: Specify `pdb_chain_id` for multi-chain PDB files

## Implementation Completeness

✅ **Full Pipeline Integration**: PDB parameters flow correctly through all stages:
1. YAML parsing → RgGuidanceConfig creation ✅
2. JSON serialization in main.py ✅ 
3. JSON deserialization in inferencev2.py ✅
4. Target calculation in potential wrappers ✅
5. Error handling and fallbacks ✅

✅ **Framework Consistency**: Rg potential integration follows identical patterns to existing potentials (SAXS, built-in potentials)

✅ **Comprehensive Testing**: All components validated with test suite

The Rg potential is now a robust, well-integrated guidance method that provides users with flexible target specification options while maintaining full consistency with the Boltz framework architecture.