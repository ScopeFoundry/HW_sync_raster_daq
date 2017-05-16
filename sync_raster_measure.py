'''

Hao Wu  Feb 4, 2015
ESB 2016-07-19
ESB 2017-02-17

'''

from ScopeFoundry.scanning import BaseRaster2DScan
from ScopeFoundry import h5_io
import numpy as np
import time
from ScopeFoundry.helper_funcs import load_qt_ui_file, sibling_path

class SyncRasterScan(BaseRaster2DScan):

    name = "sync_raster_scan"
    
    def setup(self):
        self.h_unit = self.v_unit = "V"
        self.h_limits = self.v_limits = (-10,10)
        
        BaseRaster2DScan.setup(self)
        self.Nh.update_value(1000)
        self.Nv.update_value(1000)
                
        self.display_update_period = 0.050 #seconds
        
        self.settings.New("adc_oversample", dtype=int, 
                            initial=1, 
                            vmin=1, vmax=1e10,
                            unit='x')
        self.disp_chan_choices = ['adc0', 'adc1', 'ctr0', 'ctr1'] 
        self.settings.New("display_chan", dtype=str, initial='adc0', choices=tuple(self.disp_chan_choices))        
        
        self.scanDAQ = self.app.hardware['sync_raster_daq']        
        self.scan_on=False
        
        self.details_ui = load_qt_ui_file(sibling_path(__file__, 'sync_raster_details.ui'))
        self.ui.details_groupBox.layout().addWidget(self.details_ui) # comment out?
        
        # self.ui.setWindowTitle('sync_raster_scan') #restore?
        
        self.settings.n_frames.connect_to_widget(self.details_ui.n_frames_doubleSpinBox)
        self.settings.adc_oversample.connect_to_widget(self.details_ui.adc_oversample_doubleSpinBox)
        self.settings.display_chan.connect_to_widget(self.details_ui.display_chan_comboBox)
        
        self.scanDAQ.settings.dac_rate.add_listener(self.compute_times)
        self.settings.Nh.add_listener(self.compute_times)
        self.settings.Nv.add_listener(self.compute_times)
        
        if hasattr(self.app,'sem_remcon'):#FIX re-implement later
            self.sem_remcon=self.app.sem_remcon
    
#     def dock_config(self):
#         
#         del self.ui.plot_groupBox
#         
#         
#         self.dockarea.addDock(name='SEM Sync Settings', position='left', widget=self.sem_controls)
# 
#         self.dockarea.addDock(name='Details', position='right', widget=self.details_ui)
# 
#         self.dockarea.addDock(name='Image', position='bottom', widget=self.graph_layout)
#         
# 
#     WIP
    
    def run(self):
        # if hardware is not connected, connect it
        if not self.scanDAQ.settings['connected']:
            #self.scanDAQ.connect()
            self.scanDAQ.settings['connected'] = True
            #self.app.qtapp.processEvents()
            # we need to wait while the task is created before 
            # measurement thread continues
            time.sleep(0.2)
            
        self.scanDAQ.settings['adc_oversample'] = self.settings['adc_oversample']

        # Compute data arrays        
        self.log.debug( "computing scan arrays")
        self.compute_scan_arrays()
        self.log.debug( "computing scan arrays... done")
        
        self.initial_scan_setup_plotting = True
        
        self.display_image_map = np.zeros(self.scan_shape, dtype=float)
    
    
        
        """        #Connect to RemCon and turn on External Scan for SEM
                if hasattr(self,"sem_remcon"):
                    if self.sem_remcon.connected.val:
                        self.sem_remcon.remcon.write_external_scan(1)
                   
                #self.setup_scale()
                
                if self.scanner.auto_blanking.val:
                    if hasattr(self,"sem_remcon"):
                        if self.sem_remcon.connected.val:
                            self.sem_remcon.remcon.write_beam_blanking(0)
        """                    
                        
        # previously set samples_per_point in scanDAQ hardware
               

        try:
            if self.settings['save_h5']:
                self.h5_file = h5_io.h5_base_file(self.app, measurement=self)
                self.h5_m = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5_file)
                self.display_update_period = 0.05
            else:
                self.display_update_period = 0.01

            ##### Start indexing            
            #self.frame_num = 0
            self.total_pixel_index = 0 # contains index of next adc pixel to be moved from queue into h5 file
            self.pixel_index = 0 # contains index of next adc pixel to be moved from queue into adc_pixels (within frame)
            self.current_scan_index = self.scan_index_array[0]
            self.task_done = False
            
            #### old get full image while blocking measurement thread
            #self.ai_data = self.scanDAQ.single_scan_regular(self.scan_h_positions, -1*self.scan_v_positions)
            #self.display_image_map[0,:,:] = self.ai_data[:,1].reshape(self.settings['Nv'], self.settings['Nh'])       
            ####
            
            ##### load XY positions in to DAC
            self.scanDAQ.setup_io_with_data(self.scan_h_positions, -1*self.scan_v_positions)
            
            ###### compute pixel acquisition block size 
            # need at least one, and needs to an integer divisor of Npixels
            
            num_pixels_per_block = max(1, int(np.ceil(self.display_update_period / self.scanDAQ.pixel_time)))
            if num_pixels_per_block > self.Nh.val:
                num_pixels_per_block = self.Nh.val*np.ceil( num_pixels_per_block / self.Nh.val )
    
            num_blocks = int(max(1, np.floor(self.Npixels / num_pixels_per_block)))
            
            while self.Npixels % num_blocks != 0:
                num_blocks -= 1
                #print("num_blocks", num_blocks)
        
            self.num_pixels_per_block = num_pixels_per_block = int(self.Npixels / num_blocks)
            self.log.info("num_pixels_per_block {}".format( num_pixels_per_block))
            
            ##### Data array
            # ADC
            self.adc_pixels = np.zeros((self.Npixels, self.scanDAQ.adc_chan_count), dtype=float)
            #self.adc_pixels_oversample = np.zeros((self.Npixels, self.scanDAQ.adc_chan_count*self.scanDAQ.settings['adc_oversample']))
            #self.pixels_remaining = self.Npixels # in frame
            self.new_adc_data_queue = [] # will contain numpy arrays (data blocks) from adc to be processed
            self.adc_map = np.zeros(self.scan_shape + (self.scanDAQ.adc_chan_count,), dtype=float)
            adc_chunk_size = (1,1, max(1,num_pixels_per_block/self.Nh.val), self.Nh.val ,self.scanDAQ.adc_chan_count )
            print('adc_chunk_size', adc_chunk_size)
            self.adc_map_h5 = self.create_h5_framed_dataset('adc_map', self.adc_map, chunks=adc_chunk_size, compression='gzip')
                    
            # Ctr
            # ctr_pixel_index contains index of next pixel to be processed, 
            # need one per ctr since ctrs are independent tasks
            self.ctr_pixel_index = np.zeros(self.scanDAQ.num_ctrs, dtype=int)
            self.ctr_total_pixel_index = np.zeros(self.scanDAQ.num_ctrs, dtype=int)
            self.ctr_pixels = np.zeros((self.Npixels, self.scanDAQ.num_ctrs), dtype=int)
            self.new_ctr_data_queue = [] # list will contain tuples (ctr_number, data_block) to be processed
            self.ctr_map = np.zeros(self.scan_shape + (self.scanDAQ.num_ctrs,), dtype=int)
            self.ctr_map_Hz = np.zeros(self.ctr_map.shape, dtype=float)
            ctr_chunk_size = (1,1, max(1,num_pixels_per_block/self.Nh.val), self.Nh.val, self.scanDAQ.num_ctrs)
            print('ctr_chunk_size', ctr_chunk_size)
            self.ctr_map_h5 = self.create_h5_framed_dataset('ctr_map', self.ctr_map, chunks=ctr_chunk_size, compression='gzip')
                        
            ##### register callbacks
            self.scanDAQ.set_adc_n_pixel_callback(
                num_pixels_per_block, self.every_n_callback_func_adc)
            self.scanDAQ.sync_analog_io.adc.set_done_callback(
                self.done_callback_func_adc )
            
            for ctr_i in range(self.scanDAQ.num_ctrs):
                self.scanDAQ.set_ctr_n_pixel_callback( ctr_i,
                        num_pixels_per_block, lambda i=ctr_i: self.every_n_callback_func_ctr(i))
            
            self.pre_scan_setup()

            #### Start scan daq 
            self.scanDAQ.start()
            
            #### Wait until done, while processing data queues
            while not self.task_done and not self.interrupt_measurement_called:
                self.handle_new_data()
                time.sleep(self.display_update_period)
                            
            # FIX handle serpentine scans
            #self.display_image_map[self.scan_index_array] = self.ai_data[0,:]
            # TODO save data
            

        finally:
            # When done, stop tasks
            if self.settings['save_h5']:
                self.log.info('data saved to {}'.format(self.h5_file.filename))
                self.h5_file.close()            
            self.scanDAQ.stop()
            #print("Npixels", self.Npixels, 'block size', self.num_pixels_per_block, 'num_blocks', num_blocks)
            #print("pixels remaining:", self.pixels_remaining)
            #print("blocks_per_sec",1.0/ (self.scanDAQ.pixel_time*num_pixels_per_block))
            #print("frames_per_sec",1.0/ (self.scanDAQ.pixel_time*self.Npixels))

            self.post_scan_cleanup()

        
    
    def update_display(self):
        self.get_display_pixels()
        x = self.scan_index_array.T
        self.display_image_map[x[0], x[1], x[2]] = self.display_pixels

        kk,jj, ii = self.scan_index_array[self.pixel_index]
        #self.current_stage_pos_arrow.setPos(self.h_array[ii], self.v_array[jj])
        self.current_stage_pos_arrow.setVisible(False)
        t0 = time.time()
        BaseRaster2DScan.update_display(self)
        #print("sync_raster_scan timing {}".format(time.time()-t0))
    
    ##### Callback functions
    def every_n_callback_func_adc(self):
        new_adc_data = self.scanDAQ.read_ai_chan_pixels(
            self.num_pixels_per_block)
        self.new_adc_data_queue.append(new_adc_data)
        #self.on_new_adc_data(new_data)
        num_new_pixels = new_adc_data.shape[0]
        pixel_index = self.pixel_index + num_new_pixels
        total_pixel_index =  self.total_pixel_index + num_new_pixels
        pixel_index %= self.Npixels
        if pixel_index == 0:
            frame_num = (total_pixel_index // self.Npixels) - 1
            self.on_new_frame(frame_num)
        
        return 0
    
    def every_n_callback_func_ctr(self, ctr_i):
        new_ctr_data = self.scanDAQ.read_counter_buffer(
            ctr_i, self.num_pixels_per_block)
        self.new_ctr_data_queue.append( (ctr_i, new_ctr_data))
        #print("every_n_callback_func_ctr {} {}".format(ctr_i, len(new_ctr_data)))
        return 0
            
    def done_callback_func_adc(self, status):
        self.task_done = True
        print("done", status)
        return 0
    
    def handle_new_data(self):
        while len(self.new_adc_data_queue) > 0:
            # grab the next available data chunk
            #print('new_adc_data_queue' + "[===] "*len(self.new_adc_data_queue))
            new_data = self.new_adc_data_queue.pop(0)
            self.on_new_adc_data(new_data)
            if self.interrupt_measurement_called:
                break

        while len(self.new_ctr_data_queue) > 0:
            ctr_i, new_data = self.new_ctr_data_queue.pop(0)
            self.on_new_ctr_data(ctr_i, new_data)
            if self.interrupt_measurement_called:
                break

    
    def on_new_adc_data(self, new_data):
        self.set_progress(100*self.pixel_index / self.Npixels )
        #print('callback block', self.pixel_index, new_data.shape, 'remaining px', self.Npixels - self.pixel_index)
        ii = self.pixel_index
        dii = num_new_pixels = new_data.shape[0]
        # average over samples (takes oversampled adc data and
        # gives single pixel average for each channel)
        new_data = new_data.mean(axis=2)

        #stuff into pixel data array
        self.adc_pixels[ii:ii+dii , :] = new_data
                
        self.current_scan_index = self.scan_index_array[self.pixel_index]

        self.pixel_index += num_new_pixels
        self.total_pixel_index += num_new_pixels
        
        self.pixel_index %= self.Npixels
        
        
        # copy data to image shaped map
        x = self.scan_index_array[ii:ii+dii,:].T
        self.adc_map[x[0], x[1], x[2],:] = new_data

        # Frame complete
        #pixels_remaining = self.Npixels - self.pixel_index
        #print("adc pixels_remaining", self.pixel_index, pixels_remaining, self.Npixels, frame_num)
        if self.pixel_index == 0:
            frame_num = (self.total_pixel_index // self.Npixels) - 1
            # Copy data to H5 file, if a frame is complete
            if self.settings['save_h5']:
                #print("saving h5 adc", frame_num)
                self.extend_h5_framed_dataset(self.adc_map_h5, frame_num)
                self.adc_map_h5[frame_num, :,:,:,:] = self.adc_map
                self.h5_file.flush()
            
            self.on_end_frame(frame_num - 1)
            
            # Stop scan if n_frames reached:
            if (not self.settings['continuous_scan']) \
                    and (frame_num >= self.settings['n_frames'] - 1) :
                self.task_done = True
            
    
    def on_new_frame(self, frame_i):
        pass
    
    def on_end_frame(self, frame_i):
        pass

    def on_new_ctr_data(self, ctr_i, new_data):
        #print("on_new_ctr_data {} {}".format(ctr_i, new_data))
        ii = self.ctr_pixel_index[ctr_i]
        dii = num_new_pixels = new_data.shape[0]
        
        self.ctr_pixels[ii: ii+dii, ctr_i] = new_data
        
        self.ctr_pixel_index[ctr_i] += dii
        self.ctr_total_pixel_index[ctr_i] += dii
        self.ctr_pixel_index[ctr_i] %= self.Npixels
        
        # copy data to image shaped map
        x = self.scan_index_array[ii:ii+dii,:].T
        self.ctr_map[x[0], x[1], x[2], ctr_i] = new_data
        self.ctr_map_Hz[x[0], x[1], x[2], ctr_i] = new_data *1.0/ self.scanDAQ.pixel_time

        # Frame complete
        if self.ctr_pixel_index[ctr_i] == 0:
            frame_num = (self.ctr_total_pixel_index[ctr_i] // self.Npixels) - 1
            #print('ctr frame complete', frame_num)
            # Copy data to H5 file, if a frame is complete
            if self.settings['save_h5']:
                #print('save data ctr')
                self.extend_h5_framed_dataset(self.ctr_map_h5, frame_num)
                self.ctr_map_h5[frame_num,:,:,:,ctr_i] = self.ctr_map[:,:,:,ctr_i]
                self.h5_file.flush()
        

    def pre_scan_setup(self):
        pass

    def post_scan_cleanup(self):
        pass
    
    def get_display_pixels(self):
        #DISPLAY_CHAN = 0
        #self.display_pixels = self.adc_pixels[:,DISPLAY_CHAN]
        #self.display_pixels[0] = 0
        
        chan_data = dict(
            adc0 = self.adc_pixels[:,0],
            adc1 = self.adc_pixels[:,1],
            ctr0 = self.ctr_pixels[:,0],
            ctr1 = self.ctr_pixels[:,1],
            )
        
        self.display_pixels = chan_data[self.settings['display_chan']]
        
        #self.display_pixels = self.ctr_pixels[:,DISPLAY_CHAN]
        
    def create_h5_framed_dataset(self, name, single_frame_map, **kwargs):
        """
        Create and return an empty HDF5 dataset in self.h5_m that can store
        multiple frames of single_frame_map.
        
        Must fill the dataset as frames roll in.
        
        creates reasonable defaults for compression and dtype, can be overriden 
        with**kwargs are sent directly to create_dataset
        """
        if self.settings['save_h5']:
            shape=(self.settings['n_frames'],) + single_frame_map.shape
            if self.settings['continuous_scan']:
                # allow for array to grow to store additional frames
                maxshape = (None,)+single_frame_map.shape 
            else:
                maxshape = shape
            print('maxshape', maxshape)
            default_kwargs = dict(
                name=name,
                shape=shape,
                dtype=single_frame_map.dtype,
                #chunks=(1,),
                chunks=(1,)+single_frame_map.shape,
                maxshape=maxshape,
                compression='gzip',
                #shuffle=True,
                )
            default_kwargs.update(kwargs)
            map_h5 =  self.h5_m.create_dataset(
                **default_kwargs
                )
            return map_h5
    
    def extend_h5_framed_dataset(self, map_h5, frame_num):
        """
        Adds additional frames to dataset map_h5, if frame_num 
        is too large. Adds n_frames worth of extra frames
        """
        if self.settings['continuous_scan']:
            current_num_frames, *frame_shape = map_h5.shape
            if frame_num >= current_num_frames:
                print ("extend_h5_framed_dataset", map_h5.name, map_h5.shape, frame_num)
                n_frames_extend = self.settings['n_frames']
                new_num_frames = n_frames_extend*(1 + frame_num//n_frames_extend)
                map_h5.resize((new_num_frames,) + tuple(frame_shape))
                return True
            else:
                return False
        else:
            return False
                
    def compute_times(self):
        if hasattr(self, 'scanDAQ'):
            self.settings['pixel_time'] = 1.0/self.scanDAQ.settings['dac_rate']
        BaseRaster2DScan.compute_times(self)
