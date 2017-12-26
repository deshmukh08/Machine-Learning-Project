import csv
import pandas as pd
import numpy as np
import os
def get_zone_dataframe1(filename = 'stage1_labels.csv'):
    img_id = []
    zone_no = []
    prob = []

    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            a = row[0].split('_')
            img_id.append(a[0])
            zone_no.append(a[-1])
            prob.append(row[1])

    pd_list = pd.DataFrame({
        'Id':img_id,
        'Zone':zone_no,
        'Prob':prob
        })

    return pd_list


def get_zone_threat_list1(zone='Zone5', save_as_csv=True):
    pandas_combined_list = get_zone_dataframe1()

    threat_list = pandas_combined_list.loc[(pandas_combined_list['Prob']== '1') & (pandas_combined_list['Zone']== zone)]['Id']
    non_threat_list = pandas_combined_list.loc[(pandas_combined_list['Prob']== '0') & (pandas_combined_list['Zone']== zone)]['Id']

    threat_list_train = threat_list[:int(len(threat_list)*0.8)]
    non_threat_list_train = non_threat_list[:int(len(non_threat_list)*0.8)]
    train = threat_list_train.append(non_threat_list_train)

    threat_list_test = threat_list[int(len(threat_list)*0.8):]
    non_threat_list_test = non_threat_list[int(len(non_threat_list)*0.8):]
    test = threat_list_test.append(non_threat_list_test)
    if save_as_csv:
        #threat_list.to_csv(zone + '_threat_list.csv')
        #non_threat_list.to_csv(zone + '_non_threat_list.csv')
        train.to_csv(zone + 'Train.csv')
        test.to_csv(zone + 'Test.csv')
        train_atr =[]
        with open(zone + 'Train.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                train_atr.append(row[1])


        thefile = open(zone + 'Train.csv', 'w')
        for item in train_atr:
            thefile.write("%s\n" % item)
        thefile.close()

        train_atr = []
        with open(zone + 'Test.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                train_atr.append(row[1])
        thefile = open(zone + 'Test.csv', 'w')
        for item in train_atr:
            thefile.write("%s\n" % item)
        thefile.close()
    return (threat_list, non_threat_list)

zone = get_zone_threat_list1(zone='Zone6', save_as_csv=True)
