import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Collecting results of attacks')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--method', default='all', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='all')
# parser.add_argument('--attack_details', default='cw_0.3_0.3')
parser.add_argument('--attack_details', default='pgd_0.02_0.02')
args = parser.parse_args()


if __name__ == '__main__':
    base_classifier = args.base_classifier
    seed_list = [10**i for i in range(10)]
    # seed_list = [10**i for i in range(5)]

    method = args.method
    if method == 'all':
        methods = ['lip4conv', 'nsedghi', 'gouk', 'miyato', 'orig', 'fastclip']
    elif method == 'clip':
        methods = ['lip4conv', 'nsedghi', 'gouk', 'miyato', 'fastclip']
    else:
        methods = [method]

    mode = args.mode
    if mode == 'all':
        modes = ['', 'noBN', 'clipBN_hard']
    elif mode == 'regular':
        modes = ['', 'noBN']
    elif mode == 'wBN':
        modes = ['']
    else:
        modes = [mode]

    single_df_list = []
    for mode in modes:
        for method in methods:
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            for seed in seed_list:
                # if True:
                try:
                    single_result_file = args.base_classifier + method + '_' + mode + '_' + str(int(seed)) + '/' + args.attack_details + '.csv' 
                    print(single_result_file)

                    single_df = pd.read_csv(single_result_file)
                    single_df['mode'] = mode
                    single_df['method'] = method
                    single_df['seed'] = seed   
                    single_df_list.append(single_df)

                except Exception as e:
                    print(e)
                    continue
            


    whole_df = pd.concat(single_df_list)
    whole_df = whole_df[['mode','method','acc','adv','seed']]
    whole_df.to_csv(args.base_classifier + args.attack_details + '_all_seeds.csv', index=False)

    reduced_df = whole_df[['mode','method','acc','adv']]
    df_avg = reduced_df.groupby(['mode','method']).mean()
    df_std = reduced_df.groupby(['mode','method']).std()
    print(df_avg)
    df = df_avg.merge(df_std, how='left', on=['mode', 'method'], suffixes=('_mean', '_std'))
    df.to_csv(args.base_classifier + args.attack_details + '_mean.csv', index=True)
