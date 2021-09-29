"""
Called from second cell of 'v3_dummy_beamline.py'.
This script is a slightly modified version 'sim_ptycho_scan.py'.

"""

import sys
import time
import ptypy
from ptypy import utils as u

from contrast.environment import macro, env
from contrast.recorders import active_recorders, RecorderHeader, RecorderFooter
## To load data that will be streamed:
from ptypy import io

@macro
class Dummy_Ptycho(object):
    """
    Dummy macro which streams experimental data: 'scan_000027_eiger.hdf5' and '000027.h5'.

    Does not use actual detectors or motors, but puts data in all active
    recorders.
    """

    def __init__(self):
        """
        The constructor should parse parameters.
        """

        self.scannr = env.nextScanID
        env.nextScanID += 1

        # for verbose output
        u.verbose.set_level(3) ##1

        print('Start loading data to stream')
        h5_27 = io.h5read('/Users/lexelius/Documents/Contrast/contrast-master/000027.h5')
        #scan_000027_eiger = io.h5read('/Users/lexelius/Documents/Contrast/contrast-master/scan_000027_eiger.hdf5')
        print('Finished loading data')
        """
        plt.figure()
        plt.imshow(h5_27['entry']['measurement']['eiger']['frames'][-1,:,:], vmax=1)
        plt.figure()
        plt.imshow(scan_000027_eiger['entry']['measurement']['Eiger']['data'][-1,:,:], vmax=1)
        """
        self.alldata = {}
        self.alldata['diff'] = h5_27['entry']['measurement']['eiger']['frames']##[-1, :, :]
        self.alldata['x'] = h5_27['entry']['measurement']['sx']##[-1]
        self.alldata['y'] = h5_27['entry']['measurement']['sy']##[-1]

        # # create data parameter branch
        # data = u.Param()
        # data.shape = 128##256
        # data.num_frames = 100#400
        # data.density = .2
        # data.min_frames = 1
        # data.label=None
        # data.psize=172e-6
        # data.energy= 6.2 # keV
        # data.center='fftshift'
        # data.distance = 7
        # data.auto_center = True ##None
        # data.orientation = None
        # ##data.model = 'raster' ##
        # # data.save = 'link' ##
        # # data.dfile = '/Users/lexelius/Documents/Contrast/temp/linkdata1.ptyd' ## with save='link', this creates a new .ptyd file.
        #
        # # create PtyScan instance
        # self.MF = ptypy.core.data.MoonFlowerScan(data)
        # self.MF.initialize()

        ## Proccess of     msgout = MF.auto(1) :
        ## msgout['common'] <- MF.meta
        ##      msg = MF.get_data_chunk(chunksize = frames = 1)
        ##          data, positions, weights = raw, pos, weights = MF.load(indices=indices.node) <- self._mpi_pipeline_with_dictionaries(indices)
        ##              raw[k] = intensity_j.astype(np.int32), intensity_j = u.abs2(self.geo.propagator.fw(MF.pr * MF.obj[MF.pixel[k][0]:MF.pixel[k][0] + MF.geo.shape[0], MF.pixel[k][1]:MF.pixel[k][1] + MF.geo.shape[1]]))
        ##      out = msgout = MF._make_data_package(chunk = msg)
        ## msgout['chunk'] <- msg
        ## msgout['iterable'] <-
        ##    msgout['iterable'][0]['data'] <- chunk.data.get(index)
        ##    msgout['iterable'][0]['position'] <- msg.positions
        ##    msgout['iterable'][0]['mask'] <- (msg.weights.get(index, MF.weight2d) > 0)

    def run(self):
        """
        This method does all the serious interaction with motors,
        detectors, and data recorders.
        """
        ## To Do:  check such that recorders are actually active.
        print('\nv3 Scan #%d starting at %s\n' % (self.scannr, time.asctime()))
        print('#     x          y          data')
        print('-----------------------------------------------')

        # send a header to the recorders
        snap = env.snapshot.capture()
        for r in active_recorders():
            r.queue.put(RecorderHeader(scannr=self.scannr, path=env.paths.directory,
                                       snapshot=snap, description=self._command))
        status = ''
        msgs = []
        try:
            n = 0
            while True:
                # generate the next position
                if n == self.alldata['x'].__len__():
                    stat = 'msgEOS'
                    break
                dct = {'x': self.alldata['x'][n],
                       'y': self.alldata['y'][n],
                       'diff': self.alldata['diff'][n, :, :],
                       'status' : 'running'}

                # msg = self.MF.auto(1)
                # msgs.append(msg)
                # if msg == self.MF.EOS:
                #     stat = msg
                #     break
                # d = msg['iterable'][0]
                # dct = {'x': d['position'][0],
                #        'y': d['position'][1],
                #        'diff': d['data']}

                # pass data to recorders
                for r in active_recorders():
                    r.queue.put(dct)

                # print spec-style info
                # print('%-6u%-10.4f%-10.4f%10s' % (n, dct['x']*1e6, dct['y']*1e6, dct['diff'].shape))
                print('%-6u%-10.4f%-10.4f%10s' % (n, dct['x'], dct['y'], dct['diff'].shape))

                n += 1
                time.sleep(.2) ## .2

        except KeyboardInterrupt:
            print('\nScan #%d cancelled at %s' % (self.scannr, time.asctime()))

        # tell the recorders that the scan is over
        for r in active_recorders():
            r.queue.put(RecorderFooter(scannr=self.scannr, path=env.paths.directory, status=stat))

