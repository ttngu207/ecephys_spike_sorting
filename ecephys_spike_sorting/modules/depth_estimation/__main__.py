from argschema import ArgSchemaParser
import os
import logging
import time
from pathlib import Path
import scipy.io as spio

import numpy as np

from .depth_estimation import compute_channel_offsets, find_surface_channel
from ...common.utils import write_probe_json
from ...common.SGLXMetaToCoords import MetaToCoords

def run_depth_estimation(args):

    print('ecephys spike sorting: depth estimation module\n')

    start = time.time()

    numChannels = args['ephys_params']['num_channels']

    raw_path = args['ephys_params']['ap_band_file']
    print('depth_estimation AP path is: '+ raw_path)
    rawDataAp = np.memmap(raw_path, dtype='int16', mode='r')
    dataAp = np.reshape(rawDataAp, (int(rawDataAp.size/numChannels), numChannels))

    rawDataLfp = np.memmap(args['ephys_params']['lfp_band_file'], dtype='int16', mode='r')
    raw_path = args['ephys_params']['lfp_band_file']
    print('depth_estimation LFP path is: '+ raw_path)
    dataLfp = np.reshape(rawDataLfp, (int(rawDataLfp.size/numChannels), numChannels))

    print('Computing channel offsets...')

    info_ap = compute_channel_offsets(dataAp,
                                  args['ephys_params'],
                                  args['depth_estimation_params'])

    if Path(args['ephys_params']['ap_band_file']).name.endswith('.ap.bin'):
        # SpikeGLX
        metaName, binExt = os.path.splitext(args['ephys_params']['ap_band_file'])
        metaFullPath = Path(metaName + '.meta')
        [xCoord, yCoord, shankInd] = MetaToCoords(metaFullPath, -1, badChan= np.zeros((0), dtype = 'int'), destFullPath = '', showPlot=False)
    else:
        # OpenEphys
        for d in (args['directories']['kilosort_output_directory'],
                  args['directories']['kilosort_output_tmp']):
            chanMap_path = Path(d) / 'chanMap.mat'
            if chanMap_path.exists():
                break
        else:
            chanMap_path = None

        if chanMap_path:
            chanMap = spio.loadmat(chanMap_path.as_posix(), squeeze_me=True, struct_as_record=False)
            xCoord = chanMap['xcoords']
            yCoord = chanMap['ycoords']
            shankInd = chanMap['shankInd'] - 1  # convert back to 0-based indexing
        else:
            xCoord = info_ap['horizontal_pos']
            yCoord = info_ap['vertical_pos']
            shankInd = np.full_like(xCoord, 0)  # hard-code to shank 0

    print('Computing surface channel...')

    info_lfp = find_surface_channel(dataLfp,
                                args['ephys_params'],
                                args['depth_estimation_params'],
                                xCoord,
                                yCoord,
                                shankInd)

    write_probe_json(args['common_files']['probe_json'],
                     info_ap['channels'],
                     info_ap['offsets'],
                     info_ap['scaling'],
                     info_ap['mask'],
                     info_lfp['surface_y'],
                     info_lfp['air_y'],
                     np.squeeze(yCoord),
                     np.squeeze(xCoord),
                     np.squeeze(shankInd))

    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time,2)) + ' seconds')
    print()

    return {"surface_channel": info_lfp['surface_y'],
            "air_channel": info_lfp['air_y'],
            "probe_json": args['common_files']['probe_json'],
            "execution_time": execution_time} # output manifest

def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = run_depth_estimation(mod.args)

    output.update({"input_parameters": mod.args})
    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":
    main()
