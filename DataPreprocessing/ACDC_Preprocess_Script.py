'''
@author ManalHamdi
This program preprocesses the ACDC dataset by storing temporal sequences of 2D cardiac MRI images
'''
import nibabel as nib
import numpy as np
import torch
import os
import argparse

def parse_patient_id(path):
    '''path in the format patient001/patient001_4d.nii.gz'''
    return path[7:10]

def parse_patient_id_test_success():
    path = "patient001/patient001_4d.nii.gz"
    expected_id = "001"
    actual_id = parse_patient_id(path)
    assert expected_id == actual_id, "Actual patient Id"+actual_id+"is different from the expected patient ID"+expected_id
    
def get_ACDC_temporal_seq(args, path_file, mode):
    with open(path_file) as f:
        patient_files_list = f.read().splitlines() # [[seg_frame_path1, seg_frame_path2, volume4d_path], [], ..., []]
        for line in range(0, len(patient_files_list)):
            volume_path = patient_files_list[line].split()[0]
            volume = nib.load(args.acdc_folder +"" + volume_path).get_fdata() # np array [H, W, Z, T] 
            
            H = volume.shape[0]
            W = volume.shape[1]
            Z = volume.shape[2]
            T = volume.shape[3]
            T = min(T, 19)
            if (H == 424 or W == 512):
                continue
                
            patient_id = parse_patient_id(volume_path)

            for z in range(0, Z):
                seq = np.empty([H, W, T])  #[H, W, T]
                for t in range(0, T):
                    slice_ = volume[:, :, z, t].copy()
                    seq[:,:,t] = slice_
                '''
                idx_rnd = torch.randperm(T)
                seq2 = torch.zeros_like(torch.from_numpy(seq)).numpy()
                for i in range(0, T):
                    seq2[:,:,i] = seq[:,:,idx_rnd[i]]
                
                for i in range(0, T):
                    pair = np.empty([H, W, 2])
                    pair[:,:,0] = seq[:,:,i]
                    pair[:,:,1] = seq2[:,:,i]
                    pair_filename = args.pair_folder + mode+"/pair" + str(i) +'_patient'+patient_id+'_z_'+str(z)+'.nii.gz'
                    nib_pair = nib.Nifti1Image(pair, np.eye(4))
                    nib.save(nib_pair, pair_filename)
                '''    
                nib_seq = nib.Nifti1Image(seq, np.eye(4))
                seq_filename = args.acdc_processed_folder +'/patient'+patient_id+'_z_'+str(z)+'.nii.gz'
                nib.save(nib_seq, seq_filename)
                
                if (line == 0 and z == 0): # for testing
                    volume_nib = nib.load(seq_filename).get_fdata() # Loads np array [H, W, Z, T]
                    assert volume_nib.all() == seq.all(), "The saved slices do not correspond to the loaded ones" 
                
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_folder', default='ACDC/testing', help="give the path of the folder where ACDC is stored.")
    parser.add_argument('--mode', default='testing', help="choose between testing, training and validation.")
    parser.add_argument('--acdc_processed_folder', default='ACDC_processed/testing', help="give the path of the folder where to store processed dataset.")    
    
    args = parser.parse_args()
    parse_patient_id_test_success()
    if (mode == 'training'):
        get_ACDC_temporal_seq(args, "DataPreprocessing/image_paths_training.txt", "training")
    elif (mode == 'validation'):
        get_ACDC_temporal_seq(args, "DataPreprocessing/image_paths_validation.txt", "validation")
    elif (mode == 'testing'):
        get_ACDC_temporal_seq(args, "DataPreprocessing/image_paths_testing.txt", "testing")