import csv
import pandas as pd

def get_zone_dataframe(filename = 'stage1_labels.csv'):
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

def get_zone_threat_list(zone='Zone1', save_as_csv=False):
    pandas_combined_list = get_zone_dataframe()
    threat_list = pandas_combined_list.loc[(pandas_combined_list['Prob']== '1') & (pandas_combined_list['Zone']== zone)]['Id']
    non_threat_list = pandas_combined_list.loc[(pandas_combined_list['Prob']== '0') & (pandas_combined_list['Zone']== zone)]['Id']
    if save_as_csv:
        threat_list.to_csv(zone + '_threat_list.csv')
        non_threat_list.to_csv(zone + '_non_threat_list.csv')
    return (threat_list, non_threat_list)
get_zone_threat_list('Zone1', False)
