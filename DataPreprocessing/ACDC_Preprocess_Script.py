'''
@author ManalHamdi
This program preprocesses the ACDC dataset by storing temporal sequences of 2D cardiac MRI images
'''
import nibabel as nib
import numpy as np
import os

def parse_patient_id(path):
    '''path in the format patient001/patient001_4d.nii.gz'''
    return path[7:10]

def parse_patient_id_test_success():
    path = "patient001/patient001_4d.nii.gz"
    expected_id = "001"
    actual_id = parse_patient_id(path)
    assert expected_id == actual_id, "Actual patient Id"+actual_id+"is different from the expected patient ID"+expected_id
    
def get_ACDC_temporal_seq(path_file, mode):
    with open(path_file) as f:
        patient_files_list = f.read().splitlines() # [[seg_frame_path1, seg_frame_path2, volume4d_path], [], ..., []]
        for line in range(0, len(patient_files_list)):
            volume_path = patient_files_list[line].split()[2]
            volume = nib.load("../datasets/ACDC/training/" + volume_path).get_fdata() # np array [H, W, Z, T] 
            H = volume.shape[0]
            W = volume.shape[1]
            Z = volume.shape[2]
            T = volume.shape[3]

            patient_id = parse_patient_id(volume_path)

            for z in range(0, Z):
                seq = np.empty([H, W, T])  #[H, W, T]
                for t in range(0, T):
                    slice_ = volume[:, :, z, t].copy()
                    seq[:,:,t] = slice_
                nib_seq = nib.Nifti1Image(seq, np.eye(4))
                seq_filename = '../datasets/ACDC_processed/'+mode+'/patient'+patient_id+'_z_'+str(z)+'.nii.gz'
                nib.save(nib_seq, seq_filename)

                if (line == 0 and z == 0): # for testing
                    volume_nib = nib.load(seq_filename).get_fdata() # Loads np array [H, W, Z, T]
                    assert volume_nib.all() == seq.all(), "The saved slices do not correspond to the loaded ones" 

parse_patient_id_test_success()
get_ACDC_temporal_seq("image_paths_training.txt", "training")
get_ACDC_temporal_seq("image_paths_training.txt", "validation")
get_ACDC_temporal_seq("image_paths_training.txt", "testing")