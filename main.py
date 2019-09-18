import os
import argparse
import yaml
import numpy as np
import pandas as pd
import glob
import pyproj

import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling

from shapely.geometry import Polygon
from shapely.ops import transform


def get_args():
    parser = argparse.ArgumentParser(
        description = 'Module for creating productive demand time series using irrigibility map')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads productive demand time series parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v


    #args.create_model_function = eval(args.create_model_function)
    return args

def load_regions_shapefile(args):
    shp_filename = os.path.join(args.data_dir, 'region_shapefiles', 'country_Theissen_Inters.shp')
    regions_shp = fiona.open(shp_filename)
    return regions_shp

def load_eth_shapefile(args):
    shp_filename = os.path.join(args.data_dir, 'eth_shapefile', 'ETH_outline.shp')
    regions_shp = fiona.open(shp_filename)
    return regions_shp

def load_irrigibility(args):
    '''
    Irrigation map is at 1km resolution
    '''

    filename = os.path.join(args.data_dir, 'worqlul_irrigibility', 'land_suitability.tif')

    ## Determine whether you want to save updated irrigibility map
    save_cropped = True

    ## Load irrigibility tif
    with rasterio.open(filename, 'r+') as src:
        ## Mask based on the irrigibility threshold
        cropped_src = src.read() >= args.irrigibility_lb
        cropped_src = np.array(cropped_src, dtype = np.uint8)

        metadata = src.meta
        save_image_dest = os.path.join(args.data_dir, 'worqlul_irrigibility', 'land_suitability_cropped.tif')

        if save_cropped:
            with rasterio.open(save_image_dest, 'w+', **metadata) as dest:
                dest.write(cropped_src)

        return save_image_dest

def rescale_reprojected_rainfall():
    '''
    This rescaling method can change -- my best guess at what it could be
    '''

    original    = glob.glob(args.tmp_dir + '/eth.tif')[0]
    reprojected = glob.glob(args.tmp_dir + '/eth_reproject.tif')[0]

    with rasterio.open(original, 'r') as orig:
        with rasterio.open(reprojected, 'r') as reproj:
            orig_min, orig_max     = np.min(orig.read()),   np.max(orig.read())
            reproj_min, reproj_max = np.min(reproj.read()), np.max(reproj.read())

            meta = reproj.meta

            if orig_max > reproj_max:
                reproj_array = np.maximum(reproj.read(), 0) * orig_max/reproj_max
            else:
                reproj_array = np.maximum(reproj.read(), 0) * reproj_max / orig_max


    os.remove(reprojected)

    with rasterio.open(reprojected, 'w+', **meta) as new_reproj:
        new_reproj.write(reproj_array)


def reproject_chirps_rainfall(args, chirps_filename, irrigibility_filename, eth_shp):
    '''
    CHIRPS data is at 0.05 degree spatial resolution (~5.5 km)
    Reproject to 1000m resolution
    Can also change the reprojection method
    '''

    eth_poly = Polygon(eth_shp[0]['geometry']['coordinates'][0])

    with rasterio.open(irrigibility_filename, 'r+') as irrig_src:

        dst_crs = irrig_src.meta['crs']

        ## Open CHIRPS rainfall file and crop to Ethiopia
        with rasterio.open(chirps_filename, 'r+') as chirps_src:

            out_rainfall_image, out_rainfall_transform = mask(chirps_src, [eth_poly], crop=True)
            cropped_chirps_metadata = chirps_src.meta
            out_rainfall_height = out_rainfall_image.shape[1]
            out_rainfall_width = out_rainfall_image.shape[2]
            cropped_chirps_metadata.update({"transform": out_rainfall_transform,
                             "height": out_rainfall_height,
                             "width": out_rainfall_width,
                             "crs": chirps_src.crs
                             })

            ## Save cropped Ethiopia CHIRPS data
            tmp_save_low_res = os.path.join(args.data_dir, 'chirps', 'tmp', 'eth.tif')
            with rasterio.open(tmp_save_low_res, 'w', **cropped_chirps_metadata) as eth_chirps:
                eth_chirps.write(out_rainfall_image)
                # os.remove(tmp_save_low_res)

                kwargs = irrig_src.meta.copy()
                kwargs['dtype'] = chirps_src.meta['dtype']
                tmp_save = tmp_save_low_res[0:-4] + '_reproject.tif'

                ## Reproject and upsample Ethiopia CHIRPS data
                with rasterio.open(tmp_save, 'w+', **kwargs) as reproj_dst:
                    reproject(
                        source        = rasterio.band(eth_chirps, 1),
                        destination   = rasterio.band(reproj_dst, 1),
                        resampling    = Resampling.cubic_spline)


        if args.rescale_reprojected:
            rescale_reprojected_rainfall()


def project_wsg_shape_to_csr(shape, csr):
    project = lambda x, y: pyproj.transform(
    pyproj.Proj(init='epsg:4326'),
    pyproj.Proj(init=csr),
    x, y)
    return transform(project, shape)


def calculate_energy_deficit(args, cropped_irrig_filename, regions_shp):
    chirps_reproj = glob.glob(args.tmp_dir + '/eth_reproject.tif')[0]

    num_regions = len(regions_shp)
    regional_energy_deficit = np.zeros((num_regions)) ## NOTE: For now, this is given in GWh

    with rasterio.open(cropped_irrig_filename, 'r+') as irrig_src:
        with rasterio.open(chirps_reproj, 'r+') as chirps_src:
            for i, j in enumerate(regions_shp):

                region_poly = Polygon(j['geometry']['coordinates'][0])
                projected_region = project_wsg_shape_to_csr(region_poly, 'epsg:32637')

                # Crop reprojected CHIRPS and irrigibility map
                out_chirps_image, out_chirps_transform = mask(chirps_src, [projected_region], crop=True)
                out_irrig_image,  out_irrig_transform  = mask(irrig_src,  [projected_region], crop=True)

                # Count the number of pixels that are irrigible and get indices
                irrig_pixels_indices = np.nonzero(out_irrig_image)

                # Find rainfall data at these pixels
                rainfall_overlap = out_chirps_image[irrig_pixels_indices]

                # Calculate water deficit at pixelwise basis
                water_deficit = np.maximum(args.h20_req - rainfall_overlap, 0) # mm

                # Calculate volume of water deficit, each pixel is 1000m x 1000m
                total_water_deficit_m3 = 1e-3 * np.sum(water_deficit) * 1e6

                # Calculate total energy required for water deficit, density = 1000 kg/m3, convert to GWh
                total_daily_energy = (1000 * total_water_deficit_m3 * 9.81 * args.gw_depth * args.frac_irrig *
                                      1/args.w2w_eff) / (1e9 * 3600)
                # total_avg_power = total_daily_energy / 24 # GW


                regional_energy_deficit[i] = total_daily_energy

                ## Can use this code to visualize areas with energy/water deficits
                # meta = chirps_src.meta
                # height = out_chirps_image.shape[1]
                # width = out_chirps_image.shape[2]
                # meta.update({"transform": out_chirps_transform,
                #              "height": height,
                #              "width": width,
                #              "crs": chirps_src.crs
                #              })
                #
                # temp_name = chirps_reproj[0:-4] + '_region' + str(i) + '.tif'
                # with rasterio.open(temp_name, 'w', **meta) as dest:
                #     dest.write(array_overlap)

    return regional_energy_deficit


if __name__ == '__main__':
    # get arguments from .yaml
    args = get_args()

    # print('a')

    cropped_irrig_filename = load_irrigibility(args)

    eth_shp = load_eth_shapefile(args)
    regions_shp = load_regions_shapefile(args)
    cities = [i['properties']['CityName'] for i in regions_shp]
    chirps_list = []


    chirps_dirname = os.path.join(args.chirps_dir, str(args.year))
    chirps_list = sorted(glob.glob(chirps_dirname + '/*.tif'))
    exports_df = pd.DataFrame(index = range(365), columns=cities)


    total_energy_deficit_timeseries = np.zeros((len(chirps_list), len(regions_shp)))

    for i, chirps_filename in enumerate(chirps_list):
        print(i)
        reproject_chirps_rainfall(args, chirps_filename, cropped_irrig_filename, eth_shp)
        regional_energy_deficit = calculate_energy_deficit(args, cropped_irrig_filename, regions_shp)
        exports_df.iloc[i, : ] = regional_energy_deficit

        # Clean up tmp dir
        tmp_files = glob.glob(args.tmp_dir + '/*')
        for f in tmp_files:
            os.remove(f)

    out_file = 'irrig_elec_results_year_{}_irriglb_{}_h20req_{}.csv'.format(args.year,
                                                                            args.irrigibility_lb, args.h20_req)
    outpath = os.path.join(args.elec_results_dir, out_file)
    exports_df.to_csv(outpath)

