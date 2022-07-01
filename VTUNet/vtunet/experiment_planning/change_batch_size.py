from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == '__main__':
    input_file = '/home/xychen/new_transformer/vtunetFrame/DATASET/vtunet_preprocessed/Task008_Verse1/vtunetPlansv2.1_plans_3D.pkl'
    output_file = '/home/xychen/new_transformer/vtunetFrame/DATASET/vtunet_preprocessed/Task008_Verse1/vtunetPlansv2.1_plans_3D.pkl'
    a = load_pickle(input_file)
    # a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['batch_size'] = 4
    save_pickle(a, output_file)
