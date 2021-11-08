# FEHM_supplementary
Current Version 1.0

Scripts to use LaGriT to prepare the neccesary run files for FEHM from pre-built surfaces

The class is currently built to take an externally generated surface and use it as the top of an oceanic crust layer.  The class will then populate an entire model space, generating node points, and using LaGriT to build connections.  It will then build zone files for material parameters and boundaries.  Using these, it will then generate the necessary input files for FEHM, setting you up for a super-sick modelling experience!

### Current functions:

- ***create_FEHM_run(file_prefix,parameter_file):*** initialize the class
- ***build_surfaces_real(surface_filename):*** build surfaces and node points
- ***plot_upper_surface():*** show an image of the crust top for QC purposes
- ***plot_lower_surface():*** show an image of the crust bottom for QC purposes
- ***save_coords_csv():*** export a csv with the XYZ's for the mesh nodes.  This file gets used in LaGriT but can also be used for QC purposes
- ***read_boundary_file(boundary_filename):*** read in the file with the boundary points between oceanic and continental crust.
- ***build_zones():*** build the material and boundary zone files used in FEHM run
- ***run_lagrit():*** run LaGriT to build mesh node connections
- ***write_input_file():*** build the input file for the FEHM run
- ***write_control_file():*** build the control file for the FEHM run

