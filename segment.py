from __future__ import print_function
from __future__ import division
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.misc
from scipy import ndimage
import matplotlib
from numpy import array
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
#from scipy.stats import threshold
from csv_processing import get_zone_dataframe
import sys


# GLOBALS
APS_FILE_NAME = '00360f79fd6e02781457eda48f85da90'
THREAT_LABELS = 'stage1_labels.csv'
COLORMAP = 'gray'
ZONES = 17

# Divide the available space on an image into 16 sectors. In the [0] image these
# zones correspond to the TSA threat zones.  But on rotated images, the slice
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0, 160], [200, 160], [200, 230], [0, 230]], np.int32)
sector02_pts = np.array([[0, 0], [200, 0], [200, 160], [0, 160]], np.int32)
sector03_pts = np.array([[330, 160], [512, 160], [512, 240], [330, 240]], np.int32)
sector04_pts = np.array([[350, 0], [512, 0], [512, 160], [350, 160]], np.int32)

# sector 5 is used for both threat zone 5 and 17
sector05_pts = np.array([[0, 220], [512, 220], [512, 300], [0, 300]], np.int32)

sector06_pts = np.array([[0, 300], [256, 300], [256, 360], [0, 360]], np.int32)
sector07_pts = np.array([[256, 300], [512, 300], [512, 360], [256, 360]], np.int32)
sector08_pts = np.array([[0, 370], [225, 370], [225, 450], [0, 450]], np.int32)
sector09_pts = np.array([[225, 370], [275, 370], [275, 450], [225, 450]], np.int32)
sector10_pts = np.array([[275, 370], [512, 370], [512, 450], [275, 450]], np.int32)
sector11_pts = np.array([[0, 450], [256, 450], [256, 525], [0, 525]], np.int32)
sector12_pts = np.array([[256, 450], [512, 450], [512, 525], [256, 525]], np.int32)
sector13_pts = np.array([[0, 525], [256, 525], [256, 600], [0, 600]], np.int32)
sector14_pts = np.array([[256, 525], [512, 525], [512, 600], [256, 600]], np.int32)
sector15_pts = np.array([[0, 600], [256, 600], [256, 660], [0, 660]], np.int32)
sector16_pts = np.array([[256, 600], [512, 600], [512, 660], [256, 660]], np.int32)

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_slice_list = [[  # threat zone 1
    sector01_pts, sector01_pts, sector01_pts, None,
    None, None, sector03_pts, sector03_pts,
    sector03_pts, sector03_pts, sector03_pts,
    None, None, sector01_pts, sector01_pts, sector01_pts],

    [  # threat zone 2
        sector02_pts, sector02_pts, sector02_pts, None,
        None, None, sector04_pts, sector04_pts,
        sector04_pts, sector04_pts, sector04_pts, None,
        None, sector02_pts, sector02_pts, sector02_pts],

    [  # threat zone 3
        sector03_pts, sector03_pts, sector03_pts, sector03_pts,
        None, None, sector01_pts, sector01_pts,
        sector01_pts, sector01_pts, sector01_pts, sector01_pts,
        None, None, sector03_pts, sector03_pts],

    [  # threat zone 4
        sector04_pts, sector04_pts, sector04_pts, sector04_pts,
        None, None, sector02_pts, sector02_pts,
        sector02_pts, sector02_pts, sector02_pts, sector02_pts,
        None, None, sector04_pts, sector04_pts],

    [  # threat zone 5
        sector05_pts, sector05_pts, sector05_pts, sector05_pts,
        sector05_pts, sector05_pts, sector05_pts, sector05_pts,
        None, None, None, None,
        None, None, None, None],

    [  # threat zone 6
        sector06_pts, None, None, None,
        None, None, None, None,
        sector07_pts, sector07_pts, sector06_pts, sector06_pts,
        sector06_pts, sector06_pts, sector06_pts, sector06_pts],

    [  # threat zone 7
        sector07_pts, sector07_pts, sector07_pts, sector07_pts,
        sector07_pts, sector07_pts, sector07_pts, sector07_pts,
        sector06_pts, None, None, None,
        None, None, None, None],

    [  # threat zone 8
        sector08_pts, sector08_pts, None, None,
        None, None, None, sector10_pts,
        sector10_pts, sector10_pts, sector10_pts, sector10_pts,
        sector08_pts, sector08_pts, sector08_pts, sector08_pts],

    [  # threat zone 9
        sector09_pts, sector09_pts, sector08_pts, sector08_pts,
        sector08_pts, None, None, None,
        sector09_pts, sector09_pts, None, None,
        None, None, sector10_pts, sector09_pts],

    [  # threat zone 10
        sector10_pts, sector10_pts, sector10_pts, sector10_pts,
        sector10_pts, sector08_pts, sector10_pts, None,
        None, None, None, None,
        None, None, None, sector10_pts],

    [  # threat zone 11
        sector11_pts, sector11_pts, sector11_pts, sector11_pts,
        None, None, sector12_pts, sector12_pts,
        sector12_pts, sector12_pts, sector12_pts, None,
        sector11_pts, sector11_pts, sector11_pts, sector11_pts],

    [  # threat zone 12
        sector12_pts, sector12_pts, sector12_pts, sector12_pts,
        sector12_pts, sector11_pts, sector11_pts, sector11_pts,
        sector11_pts, sector11_pts, sector11_pts, None,
        None, sector12_pts, sector12_pts, sector12_pts],

    [  # threat zone 13
        sector13_pts, sector13_pts, sector13_pts, sector13_pts,
        None, None, sector14_pts, sector14_pts,
        sector14_pts, sector14_pts, sector14_pts, None,
        sector13_pts, sector13_pts, sector13_pts, sector13_pts],

    [  # sector 14
        sector14_pts, sector14_pts, sector14_pts, sector14_pts,
        sector14_pts, None, sector13_pts, sector13_pts,
        sector13_pts, sector13_pts, sector13_pts, None,
        None, None, None, None],

    [  # threat zone 15
        sector15_pts, sector15_pts, sector15_pts, sector15_pts,
        None, None, sector16_pts, sector16_pts,
        sector16_pts, sector16_pts, None, sector15_pts,
        sector15_pts, None, sector15_pts, sector15_pts],

    [  # threat zone 16
        sector16_pts, sector16_pts, sector16_pts, sector16_pts,
        sector16_pts, sector16_pts, sector15_pts, sector15_pts,
        sector15_pts, sector15_pts, sector15_pts, None,
        None, None, sector16_pts, sector16_pts],

    [  # threat zone 17
        None, None, None, None,
        None, None, None, None,
        sector05_pts, sector05_pts, sector05_pts, sector05_pts,
        sector05_pts, sector05_pts, sector05_pts, sector05_pts]]

# ----------------------------------------------------------------------------------
# read_header(infile):  takes an aps file and creates a dict of the data
#
# infile:               an aps file
#
# returns:              all of the fields in the header
# ----------------------------------------------------------------------------------
def read_header(APS_FILE_NAME):
    # declare dictionary
    h = dict()

    with open(APS_FILE_NAME, 'r+b') as fid:
        h['filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['energy_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['config_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['file_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['trans_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype='S1', count=16))
        h['frequency'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['mat_velocity'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare00'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['band_width'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare01'] = np.fromfile(fid, dtype=np.int16, count=5)
        h['polarization_type'] = np.fromfile(fid, dtype=np.int16, count=4)
        h['record_header_size'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_precision'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['min_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['max_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['avg_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_scale_factor'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_units'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['surf_removal'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['edge_weighting'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['x_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['y_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['z_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['t_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['spare02'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['scan_orientation'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_direction'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_storage_order'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_x_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_y_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_z_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_t_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['x_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['depth_recon'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare06'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_radius'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_delay'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_start'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_end'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['ahis_software_version'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare_end'] = np.fromfile(fid, dtype=np.float32, count=10)

    return h


# ----------------------------------------------------------------------------------
# read_data(infile):  reads and rescales any of the four image types
#
# infile:             an .aps, .aps3d, .a3d, or ahi file
#
# returns:            the stack of images
#
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16
# ----------------------------------------------------------------------------------
def read_data(infile):
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])

    extension = os.path.splitext(infile)[1]

    with open(infile, 'rb') as fid:

        # skip the header
        fid.seek(512)

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':

            if (h['word_type'] == 7):
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)

            elif (h['word_type'] == 4):
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':

            if (h['word_type'] == 7):
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)

            elif (h['word_type'] == 4):
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy()

            # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0, :, :, :].copy()
            imag = data[1, :, :, :].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag


# ----------------------------------------------------------------------------------
# get_single_image(infile, nth_image):  returns the nth image from the image stack
#
# infile:                              an aps file
#
# returns:                             an image
# ----------------------------------------------------------------------------------
def get_single_image(infile, nth_image):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    return np.flipud(img[nth_image])


#----------------------------------------------------------------------------------
# convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------
def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)
    return np.uint8(img_rescaled)

def print_header(header):
    print('{:16}{}'.format('Key', 'Value'))
    for data_item in sorted(header):
        print ('{:15}:{}'.format(data_item, header[data_item]))
    print('')

def get_crop_dimensions(angle, zone):
    threat_zone = zone_slice_list[zone - 1]
    #print('threat_zone_1',threat_zone)

    sector_pts = threat_zone[angle - 1]
    #print('sector_pts', sector_pts)

    if sector_pts is not None:
        sector_pts_00 = sector_pts[:, 0]
        sector_pts_01 = sector_pts[:, 1]
        min_0 = min(sector_pts_00)
        max_0 = max(sector_pts_00)
        min_1 = min(sector_pts_01)
        max_1 = max(sector_pts_01)
        crop_zone = [min_1, max_1, min_0, max_0]
        return crop_zone
    else:
        return None

# Use this function for batch processing, see test() for example
# No Thresholding done yet
def get_cropped_zones(data_dir, filelist, file_extension, angle):
    zones = []
    #zones.append((None, None))

    for zone in range(1, ZONES + 1):
        #tuple (zoned images, zoned lables)
        zonedImage = ([], [])
        zones.append(zonedImage)

    # [Id, Zone, Prob]
    df = get_zone_dataframe()

    #filter on filelist
    df = df[df['Id'].isin(filelist)]
    i=1;
    for filename in filelist:
        file_df = df[df['Id'] == filename]
        single_image = get_single_image(data_dir + filename + '.' + file_extension, angle-1) #angle needs to subracted with to normalze with get_crop_dimenisons function
        single_image = convert_to_grayscale(single_image)
        for zone in range(1, ZONES + 1):
            #print('Runnning zone:' + str(zone) + ' angle:' + str(angle))

            crop_dim = get_crop_dimensions(angle, zone)
            if crop_dim is not None:
                cropped_zone = single_image[crop_dim[0]:crop_dim[1],crop_dim[2]:crop_dim[3]]
                zones[zone-1][0].append(cropped_zone)
                label = file_df[file_df['Zone'] == 'Zone' + str(zone)]['Prob']
                zones[zone-1][1].append([int(label.values[0])])
            else:
                a=1
                #print('Zone ' + str(zone) + ' not available for ' + filename + ' at angle ' + str(angle))
        print('Percentage Complete - ' + "{0:.2f}".format(i*100/len(filelist)) + "%\r",end="")
        #sys.stdout.flush()
        i=i+1
    print("")
    return zones


def test():

    # setup
    angle = 8
    zone = 1
    COLORMAP = 'gray'
    filename = './' + APS_FILE_NAME + '.aps'

    try:
        for f in os.listdir('./test'):
            os.remove(os.path.join('./test/', f))
        os.rmdir('./test')
    except OSError:
        pass
    except Exception as e:
        print(e)

    try:
        os.mkdir('./test')
    except OSError:
        pass
    except Exception as e:
        print(e)

    # test
    print('Printing headers:')
    print_header(read_header(filename))

    # test
    print('Reading image data')
    read_data(filename)

    # test
    print('converting to grayscale')
    grayscale = convert_to_grayscale(get_single_image(filename, angle))
    plt.imshow(grayscale, cmap = plt.get_cmap(COLORMAP))
    plt.savefig('./test/full_image', dpi=100)

    # test
    print('Getting zones information')
    zones = get_cropped_zones('./', filelist = [APS_FILE_NAME], file_extension='aps', angle=angle)

    for zone in range(1, ZONES + 1):
        print ('Saving zone ' + str(zone) + ' image')
        images, labels = zones[zone]
        if len(images) > 0:
            plt.imshow(images[0], cmap = plt.get_cmap(COLORMAP))
            plt.savefig('./test/img_zone-' + str(zone) + '_label-' + str(labels[0]), dpi=100)
        else:
            print('No images present in this zone!')

#--matrix--
# np.set_printoptions(precision=3,)
# print(*crop)
# print(type(crop))
# thresholded = threshold(crop, 50)
# #np.savetxt('test1.txt', crop_zone_13, fmt='%d')
# np.savetxt('test_with_object.txt', thresholded, fmt='%d', newline='\n' )

#if __name__ == '__main__':
    #print('Executing test')
    #test()
