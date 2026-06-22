import os
import mdtraj as md

def create_dataset(files, output_file, snapshot_gap):
    with open(output_file, 'w') as f_out:
        f_out.write("Snapshot_i,Snapshot_i+1\n")

        for file in files:
            if os.path.exists(file):
                # Load the trajectory
                traj = md.load(file, top='C:/Users/518408/Desktop/Phd project/RLuc8B/filtered.pdb')

                # Select C-alpha atoms
                ca_indices = traj.top.select('name CA')
                ca_traj = traj.atom_slice(ca_indices)

                # Sample consecutive snapshots with the specified gap
                num_snapshots = 0
                for i in range(0, len(ca_traj) - snapshot_gap, snapshot_gap):
                    snapshot_i = ca_traj[i]
                    snapshot_i1 = ca_traj[i + snapshot_gap]

                    # Write the C-alpha coordinates to the output file
                    snapshot_i_coords = ",".join(map(str, snapshot_i.xyz.flatten()))
                    snapshot_i1_coords = ",".join(map(str, snapshot_i1.xyz.flatten()))
                    f_out.write(f"{snapshot_i_coords},{snapshot_i1_coords}\n")
                    num_snapshots += 1

                print(f"Extracted {num_snapshots} snapshots from {file}")

# Specify the directory containing the adaptive sampling files
directory_path = 'C:/Users/518408/Desktop/Phd project/RLuc8B/filtered_aligned'

# Initialize an empty list to store the XTC files
xtc_files = []

# Print all files in the directory and store XTC files in the list
print("XTC files in the directory:")
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        if file_name == "output.filtered.align.xtc":
            xtc_files.append(os.path.join(root, file_name))
            print(file_name)

# Specify the output file for the dataset
output_file = "dataset.csv"

# Specify the snapshot gap between consecutive snapshots
snapshot_gap = 50

# Create the dataset
create_dataset(xtc_files, output_file, snapshot_gap)
