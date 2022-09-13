import os
import argparse

def GenerateFile(args):
    f = open(args.file_full_path, "w") #seg_frame1 seg_frame2 volume_4d

    for patient_folder_name in os.listdir(args.acdc_folder):
        line = []
        if (patient_folder_name == "patient001.Info.cfg"):
            continue
        for frame_name in os.listdir(args.acdc_folder + patient_folder_name):
            if (frame_name == "Info.cfg"):
                continue
            else:
                line.append(patient_folder_name + "/" + frame_name)
        line.sort()
        f.write(" ".join(line))
        f.write("\n")

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_folder', default='raft', help="give the path of the folder where ACDC is stored.")
    parser.add_argument('--file_full_path', default='raft', help="give the path of file where to write the name of the patient data.")    
    args = parser.parse_args()
    GenerateFile(args)