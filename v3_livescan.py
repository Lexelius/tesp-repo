"""
To be run before second cell- and after first cell of 'v3_dummy_beamline.py'.

----------------------------------------------------------------------------
Things to fix / troubleshooting notes:
----------------------------------------------------------------------------
* If num_iter have been reached before all patterns have been collected then
    Ptycho stops because it thinks it's finished!
    -> Define a number of iterations for recon that starts counting only after scan is over: yes!
* If 'p.scans.contrast.data.shape = 128' is included in the livescan.py script then
    ptycho-reconstruction becomes super fast and stops before scan is over as above.
* Check: how to specify " kind = 'full_flat' " as input for save_run(..) that is called by Ptycho.
    Or rather, is there a way to save the pods to the .ptyr files as well?
* Check if there is a way to see how many pods/frames that are included in each .ptyr file.
* Ptycho keeps going in to LiveScan.check() after all frames have been acquired, which overwrites
    then self.end_of_scan..

* Number of frames included in iteration  0    10    20    30  40  50  60  70  80  90  100
    min_frames = 10, DM.numiter = 10:
        check return:                     6*   8*9    0*
        latest_pos_index_received         27* 69*105 120*
        Repackaged data from frame        21* 61*96  120*
        .ptyr / error_local                -   21    96   120

    min_frames = 1, DM.numiter = 1:
        check return:                     0     0
        latest_pos_index_received         28    120
        Repackaged data from frame        28    120
        .ptyr / error_local               -     120
    min_frames = 1, DM.numiter = 10:
        check return:                     0     0
        latest_pos_index_received         29    120
        Repackaged data from frame        29    120
        .ptyr / error_local

 """

"""
Notes:
-----------------------------------------------------------
Subclasses of PtyScan can be made to override to tweak the methods of base class PtyScan.
Methods defined in PtyScan(object) are:
    ¨def __init__(self, pars=None, **kwargs):
    def initialize(self):
    ¨def _finalize(self):
    ^def load_weight(self):
    ^def load_positions(self):
    ^def load_common(self):
    def post_initialize(self):
    ¨def _mpi_check(self, chunksize, start=None):
    ¨def _mpi_indices(self, start, step):
    def get_data_chunk(self, chunksize, start=None):
    def auto(self, frames):
    ¨def _make_data_package(self, chunk):
    ¨def _mpi_pipeline_with_dictionaries(self, indices):
    ^def check(self, frames=None, start=None):
    ^def load(self, indices):
    ^def correct(self, raw, weights, common):
    ¨def _mpi_autocenter(self, data, weights):
    def report(self, what=None, shout=True):
    ¨def _mpi_save_chunk(self, kind='link', chunk=None):

¨: Method is protected (or private if prefix is __).
^: Description explicitly says **Override in subclass for custom implementation**.
"""

import numpy as np
import zmq
from zmq.utils import jsonapi as json
import time
import bitshuffle
import struct
from ptypy.core import Ptycho
from ptypy.core.data import PtyScan
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.experiment import register
from ptypy.utils.verbose import headerline


logger = u.verbose.logger
def logger_info(*arg):
    """
    Just an alternative to commenting away logger messages.
    """
    return

##@defaults_tree.parse_doc('scandata.LiveScan')
@register()
class LiveScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = LiveScan
    help =

    [host]
    default = '127.0.0.1'
    type = str
    help = Name of the publishing host
    doc =

    [port]
    default = 5556
    type = int
    help = Port number on the publishing host
    doc =
    """

    def __init__(self, pars=None, **kwargs):

        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        p = self.DEFAULT.copy(depth=99)
        p.update(pars)
        p.update(kwargs)

        super(LiveScan, self).__init__(p, **kwargs)
        self.context = zmq.Context()

        # main socket
        socket = self.context.socket(zmq.SUB)
        socket.connect("tcp://%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.socket = socket

        self.latest_pos_index_received = -1
        self.incoming = {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')


    def check(self, frames=None, start=None):
        """
        Only called on the master node.
        """
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        self.end_of_scan = False

        # get all frames from the main socket
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)  ## NOBLOCK returns None if a message is not ready
                logger.info('######## Received a message')  ##
                ##headers = ('path' in msg.keys())
                ##emptymsg = ('heartbeat' in msg.values()) # just a message from contrast to keep connection alive
                ######if 'running' in msg.values():  # if zmq did not send a path: then save this message
                if msg['status'] == 'running':
                    self.latest_pos_index_received += 1
                    self.incoming[self.latest_pos_index_received] = msg
                    logger.info('############ Frame nr. %d received' % self.latest_pos_index_received)  ##
                elif msg['status'] == 'msgEOS':  # self.EOS:
                    self.end_of_scan = True
                    logger.info('############ RecorderFooter received; END OF SCAN!')  ##
                    break
                else:
                    logger.info('############ Message was not important')  ##
            except zmq.ZMQError:
                logger.info('######## Waiting for messages')  ##
                # no more data available - working around bug in ptypy here
                if self.latest_pos_index_received < self.info.min_frames * parallel.size:
                    logger.info('############ self.latest_pos_index_received = %u , self.info.min_frames = %d , parallel.size = %d' % (self.latest_pos_index_received, self.info.min_frames, parallel.size))  ##
                    logger.info('############ Not enough frames received, have %u frames, waiting...' % (self.latest_pos_index_received + 1))
                    time.sleep(1)
                else:
                    logger.info('############ Will process gathered data')  ##
                    break

        ind = self.latest_pos_index_received
        logger.info('#### latest_pos_index_received = %d' % ind)
        logger.info('#### check return (ind - start + 1) = %d, self.end_of_scan = %d' % ((ind - start + 1), self.end_of_scan))
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')
        return (ind - start + 1), self.end_of_scan

    def load(self, indices):
        # indices are generated by PtyScan's _mpi_indices method.
        # It is a diffraction data index lists that determine
        # which node contains which data.
        raw, weight, pos = {}, {}, {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))

        # communication
        if parallel.master:
            # send data to each node
            for node in range(1, parallel.size):
                node_inds = parallel.receive(source=node)
                dct = {i: self.incoming[i] for i in node_inds}
                parallel.send(dct, dest=node)
                for i in node_inds:
                    del self.incoming[i]
            # take data for this node
            dct = {i: self.incoming[i] for i in indices}
            for i in indices:
                del self.incoming[i]
        else:
            # receive data from the master node
            parallel.send(indices, dest=0)
            dct = parallel.receive(source=0)
        #logger_info(dct)

        # repackage data and return
        for i in indices:
            try:
                raw[i] = dct[i]['diff']
                # raw[i] = dct[i][self.info.detector]
                # #            pos[i] = np.array([
                # #                        dct[i][self.info.xMotor],
                # #                        dct[i][self.info.yMotor],
                # #                        ]) * 1e-6
                # x = dct[i][self.info.xMotor]
                # y = dct[i][self.info.yMotor]
                x = dct[i]['x']
                y = dct[i]['y']
                #pos[i] = np.array((x, y))
                pos[i] = -np.array((y, -x)) * 1e-6
                logger_info(pos[i])
                weight[i] = np.ones_like(raw[i])
                weight[i][np.where(raw[i] == 2 ** 32 - 1)] = 0

                # d = msg['iterable'][0]
                # dct = {'x': d['position'][0],
                #        'y': d['position'][1],
                #        'diff': d['data']}
                logger.info('######## Repackaged data from frame nr. : %d' % i)
            except:
                break
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

        return raw, pos, weight


LS = LiveScan()
LS.initialize()


############################################################################
# hard coded user input
############################################################################
detector = 'eiger' # or 'merlin' or 'pilatus'
beamtime_basedir = '/Users/lexelius/Documents/Contrast/contrast-master/experimental-data'## '/data/staff/nanomax/commissioning_2021-2/20210907/'
sample = 'setup'
distance_m = 3.5	# distance between the sample and the detector in meters
defocus_um = 300	# distance between the focus and the sample plane in micro meters -> used for inital probe
energy_keV = 6.5 ## #energy_keV = 6.5	# incident photon energy in keV ... now read from scan file
scannr = 27## int(sys.argv[1])

# where to put the reconstructions
out_dir = beamtime_basedir + '/livescan_output/'
out_dir_data    = out_dir + 'data/'
out_dir_dumps   = out_dir + 'dumps/'
out_dir_rec     = out_dir + 'rec/'
# and what the files are supposed to be called
path_data       = out_dir_data  + 'data_scan_' + str(scannr).zfill(6) + '.ptyd'							    # the file with the prepared data
path_dumps      = out_dir_dumps + 'dump_scan_' + str(scannr).zfill(6)+'_%(engine)s_%(iterations)04d.ptyr'   # intermediate results
path_rec        = out_dir_rec   + 'rec_scan_' + str(scannr).zfill(6)+'_%(engine)s_%(iterations)04d.ptyr'	# final reconstructions (of each engine)

############################################################################
# creating the parameter tree
############################################################################

# General parameters
p = u.Param()
p.verbose_level = 3
##p.run = 'scan%d' % scannr

# where to put the reconstructions
p.io = u.Param()
p.io.home = out_dir_rec
p.io.autoplot = u.Param() ##
p.io.autoplot.active = False ##
p.io.rfile = path_rec
p.io.autosave = u.Param()
p.io.autosave.rfile = path_dumps

# Scan parameters
p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'Full'
p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = 1		# Number of probe modes
#p.scans.scan00.coherence.num_object_modes = 4		# Number of object modes


p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'LiveScan' ## 'NanomaxContrast'
##p.scans.scan00.data.path = beamtime_basedir+'/raw/'+sample+'/'
####p.scans.scan00.data.detector = detector ####Parameter validation failed (not defined in livescan yet)
####p.scans.scan00.data.maskfile = {'merlin': '/data/visitors/nanomax/common/masks/merlin/latest.h5', 'pilatus': None, 'eiger': None,}[detector] ####Parameter validation failed (not defined in livescan yet)
####p.scans.scan00.data.scanNumber = scannr ####Parameter validation failed (not defined in livescan yet)
##p.scans.scan00.data.xMotor = 'sx'
##p.scans.scan00.data.yMotor = 'sy'
#p.scans.scan00.data.zDetectorAngle = -0.03 # [deg]
p.scans.scan00.data.shape = 256 #256
p.scans.scan00.data.save = 'append' #'link'
p.scans.scan00.data.dfile = path_data
p.scans.scan00.data.center = (1364,626) ####None # auto, you can also set (i, j) center here.
####p.scans.scan00.data.xMotorFlipped = True ####Parameter validation failed (not defined in livescan yet)
p.scans.scan00.data.orientation = {'merlin': (False, False, True), 'pilatus': None, 'eiger': (False, True, False)}[detector]

#p.scans.scan00.data.orientation = {'merlin': (False, False, True), 'pilatus': None, 'eiger': None}[detector]
p.scans.scan00.data.distance = distance_m
p.scans.scan00.data.psize = {'pilatus': 172e-6, 'merlin': 55e-6, 'eiger': 75e-6}[detector]
p.scans.scan00.data.energy = energy_keV ## #p.scans.scan00.data.energy = energy_keV
####p.scans.scan00.data.I0 = None # can be like 'alba2/1' ####Parameter validation failed (not defined in livescan yet)
p.scans.scan00.data.min_frames = 1 ##10
p.scans.scan00.data.load_parallel = 'all'

# Scan parameters: illumination
p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = None
p.scans.scan00.illumination.aperture = u.Param()
p.scans.scan00.illumination.aperture.form = 'rect'
p.scans.scan00.illumination.aperture.size = 150e-9			           # at the focus
p.scans.scan00.illumination.propagation = u.Param()
p.scans.scan00.illumination.propagation.parallel = 1.*defocus_um*1e-6 # somehow this has to be negative to the basez axis
															           # -> being downstream of the focus means negative distance



############################################################################
# 1st difference map
############################################################################

# general
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.numiter_contiguous = 1 ##10


############################################################################
# 2nd maximum likelihood
############################################################################

if False:
    # general
    p.engines.engine01 = u.Param()
    p.engines.engine01.name = 'ML'
    p.engines.engine01.numiter = 100
    p.engines.engine01.numiter_contiguous = 10

LS = LiveScan(pars=p)
LS.initialize()

P = Ptycho(p,level=5)

# # P.plot_overview()
#
# # u.plot_storage(list(P.obj.storages.values())[0], fignum=100, modulus='linear',
# #                slices=(slice(1), slice(None), slice(None)), si_axes='x', mask=None)
# # u.plot_client.MPLplotter.create_plot_from_tile_list(fignum=1, num_shape_list=[(4, (2, 2))], figsize=(8, 8))
# # fig = u.plot_storage(list(P.diff.storages.values())[0], 0, slices=(slice(4), slice(None), slice(None)), modulus='log')
# # fig = u.plot_client.MPLplotter.plot_storage(list(P.diff.storages.values())[0], 0,
# #                                             slices=(slice(4), slice(None), slice(None)), modulus='log')
#
# #%% plots from pods
# from ptypy.utils import imsave
# import matplotlib.pyplot as plt
# # fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row',figsize=(16,4))
# fig, axs = plt.subplots(nrows=2, ncols=10, figsize=(16,4))
#
# fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
# plt.setp(axs[0,0], title='Object')
# plt.setp(axs[1,0], title='Probe')
# k=0
# axs[0,k].imshow(imsave(P.pods['P0000'].object))
# axs[1,k].imshow(imsave(P.pods['P0000'].probe))
# for i in range(10,90,10):
#     k += 1
#     axs[0,k].imshow(imsave(P.pods[f"P{i:04d}"].object))
#     axs[1,k].imshow(imsave(P.pods[f"P{i:04d}"].probe))
# k += 1
# axs[0,k].imshow(imsave(P.pods['P0091'].object))
# axs[1,k].imshow(imsave(P.pods['P0091'].probe))

# #%%  Plot probe and object sample through out reconstruction.
from ptypy import io
from ptypy.utils import imsave
import matplotlib.pyplot as plt
import numpy as np
def ptyrplot():
    """

    example:
    ptyrplot(dump_path = ''. rec_path = '')
    """
    from ptypy import io
    from ptypy.utils import imsave
    import matplotlib.pyplot as plt
    import numpy as np
    # fig, axs = plt.subplots(nrows=3, ncols=11, sharex='col', sharey='row',figsize=(16,4))
    VMAX = 1.2
    fig, axs = plt.subplots(nrows=3, ncols=11, figsize=(19,4.5))
    # plt.rcParams['xtick.labelsize'] = 6
    fig.subplots_adjust(left=0.04, bottom=0.05, right=0.999, top=0.9, wspace=0.36, hspace=0.2)
    plt.setp(axs[0, 0], ylabel='Object')
    plt.setp(axs[1, 0], ylabel='Phase')
    plt.setp(axs[2, 0], ylabel='Probe')
    plt.setp(axs[0, 0], title=f"0000")
    c_obj, c_probe = [], []
    prefix1 = path_dumps.split('%')[0]
    rfile0 = prefix1 + 'None_0000.ptyr'

    content = io.h5read(rfile0, 'content')['content']
    c_obj.append(content['obj']['Sscan00G00']['data'])
    c_probe.append(content['probe']['Sscan00G00']['data'])
    k=0
    axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
    axs[1, k].imshow(np.angle(c_obj[k][0]))
    axs[2, k].imshow(imsave(c_probe[k][0]))
    for i in range(10,100,10):
        k += 1
        filename = prefix1 + f"DM_{i:04d}.ptyr"
        content = io.h5read(filename, 'content')['content']
        c_obj.append(content['obj']['Sscan00G00']['data'])
        c_probe.append(content['probe']['Sscan00G00']['data'])
        axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
        axs[1, k].imshow(np.angle(c_obj[k][0]))
        axs[2, k].imshow(imsave(c_probe[k][0]))
        plt.setp(axs[0, k], title=f"DM_{i:04d}")
    k += 1
    filename = path_rec.split('%')[0] + f"DM_{i+10:04d}.ptyr"
    content = io.h5read(filename, 'content')['content']
    c_obj.append(content['obj']['Sscan00G00']['data'])
    c_probe.append(content['probe']['Sscan00G00']['data'])
    last_obj = axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
    last_phase = axs[1, k].imshow(np.angle(c_obj[k][0]))
    axs[2, k].imshow(imsave(c_probe[k][0]))
    plt.setp(axs[0, k], title=f"DM_{i+10:04d}")
    plt.rcParams['xtick.labelsize'] = 8 ## default= 10
    plt.rcParams['ytick.labelsize'] = 8 ## default= 10
    plt.show()

    c_pos = content['positions']['Sscan00G00']
    figpos = plt.figure(num=2, figsize=(5, 5))
    plt.plot(c_pos[:, 0], c_pos[:, 1], '-X')
    plt.title(f'Scan trajectory, {c_pos.__len__()} positions')
    plt.show()
#
# #%%  Plot probe and object sample through out reconstruction. f"pydevconsole_DM_{i:04d}.ptyr"
# from ptypy import io
# from ptypy.utils import imsave
# import matplotlib.pyplot as plt
# import numpy as np
# # fig, axs = plt.subplots(nrows=3, ncols=11, sharex='col', sharey='row',figsize=(16,4))
# VMAX = 1.2
# fig, axs = plt.subplots(nrows=3, ncols=11, figsize=(19,4.5))
# # plt.rcParams['xtick.labelsize'] = 6
# fig.subplots_adjust(left=0.04, bottom=0.05, right=0.999, top=0.9, wspace=0.36, hspace=0.2)
# plt.setp(axs[0, 0], ylabel='Object')
# plt.setp(axs[1, 0], ylabel='Phase')
# plt.setp(axs[2, 0], ylabel='Probe')
# plt.setp(axs[0, 0], title=f"0000")
# c_obj, c_probe = [], []
# rfile0 = '/Users/lexelius/Documents/Contrast/contrast-master/experimental-data/process/setup/scan_000027/ptycho_ptypy/dumps/dump_scan_000027_None_0000.ptyr'
#
# content = io.h5read(rfile0, 'content')['content']
# c_obj.append(content['obj']['Sscan00G00']['data'])
# c_probe.append(content['probe']['Sscan00G00']['data'])
# k=0
# axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
# axs[1, k].imshow(np.angle(c_obj[k][0]))
# axs[2, k].imshow(imsave(c_probe[k][0]))
# rfile10 = '/Users/lexelius/Documents/Contrast/contrast-master/experimental-data/process/setup/scan_000027/ptycho_ptypy/dumps/dump_scan_000027_ML_0110.ptyr'
# pref = rfile10.split('ML_')[0]
# suff = '.ptyr'
# for i in range(110,200,10):
#     k += 1
#     filename = pref + f"ML_{i:04d}" + suff
#     content = io.h5read(filename, 'content')['content']
#     c_obj.append(content['obj']['Sscan00G00']['data'])
#     c_probe.append(content['probe']['Sscan00G00']['data'])
#     axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
#     axs[1, k].imshow(np.angle(c_obj[k][0]))
#     axs[2, k].imshow(imsave(c_probe[k][0]))
#     plt.setp(axs[0, k], title=f"ML_{i:04d}")
# k += 1
# filename = '/Users/lexelius/Documents/Contrast/contrast-master/experimental-data/process/setup/scan_000027/ptycho_ptypy/rec/rec_scan_000027_ML_0200.ptyr'
# content = io.h5read(filename, 'content')['content']
# c_obj.append(content['obj']['Sscan00G00']['data'])
# c_probe.append(content['probe']['Sscan00G00']['data'])
# last_obj = axs[0, k].imshow(imsave(c_obj[k][0], vmax=VMAX))
# last_phase = axs[1, k].imshow(np.angle(c_obj[k][0]))
# axs[2, k].imshow(imsave(c_probe[k][0]))
# plt.setp(axs[0, k], title=f"ML_0200")
# plt.rcParams['xtick.labelsize'] = 8 ## default= 10
# plt.rcParams['ytick.labelsize'] = 8 ## default= 10
# plt.show()
#
# c_pos = content['positions']['Sscan00G00']
# figpos = plt.figure(num=2, figsize=(5, 5))
# plt.plot(c_pos[:, 0], c_pos[:, 1], '-X')
# plt.title(f'Scan trajectory, {c_pos.__len__()} positions')
# plt.show()
#
# ## plt.figure(num=3)
# ## plt.imshow(abs(c_obj[k][0]), vmax=2)
#
